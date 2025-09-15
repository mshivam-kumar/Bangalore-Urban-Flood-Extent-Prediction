import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.utils import plots
from src.models.loss import masked_bce_loss

def train_model(model, train_loader, val_loader, epochs, pad_val, nodata_dict, channel_names, model_dir="", plots_dir=""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # --- Optimizer ---
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor([1.2], device=device))
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    
    pos_weight = 1.4
    criterion = None
    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor([pos_weight], device=device))

    else:
        criterion = nn.BCEWithLogitsLoss(reduction='none')

    model = model.to(device)
    
    # Variables to track best model
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for features_dict, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            stack_lst = []
            for channel_name in channel_names:
                if channel_name == "Rainfall":  # we are handling rainfall separately below
                    continue
                stack_lst.append(features_dict[channel_name])
            
            selected_features = torch.stack(stack_lst, dim=1)
            static_features = selected_features
            rainfall = features_dict['Rainfall']

            x_static = static_features.to(device)
            x_rainfall = rainfall.to(device)
            labels = labels.to(device)

            # optimizer.zero_grad()
            optimizer.zero_grad(set_to_none=True)
            outputs = model(x_static, x_rainfall)
            outputs = outputs.squeeze(1)

            # loss = masked_bce_loss(outputs, labels, x_static, pad_val, nodata_dict, channel_names)
            loss = masked_bce_loss(outputs, labels, x_static, pad_val, nodata_dict, channel_names, criterion)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            # print(f"Loss: {train_loss}")

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)


        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features_dict, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                stack_lst = []
                for channel_name in channel_names:
                    if channel_name == "Rainfall":
                        continue
                    stack_lst.append(features_dict[channel_name])
                
                selected_features = torch.stack(stack_lst, dim=1)
                static_features = selected_features
                rainfall = features_dict['Rainfall']

                x_static = static_features.to(device)
                x_rainfall = rainfall.to(device)
                labels = labels.to(device)

                outputs = model(x_static, x_rainfall)
                outputs = outputs.squeeze(1)

                # loss = masked_bce_loss(outputs, labels, x_static, pad_val, nodata_dict, channel_names)
                loss = masked_bce_loss(outputs, labels, x_static, pad_val, nodata_dict, channel_names, criterion)

                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        history['val_loss'].append(avg_val_loss)
        print(f"Epoch {epoch}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}")
        
        if(epoch%4==0):
            plots.train_and_validation_loss_curve(epoch, train_loss=history['train_loss'], val_loss=history['val_loss'], output_dir=plots_dir)

        # Check if this is the best model so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            best_model_state = model.state_dict().copy()  # Save a copy of the state dict
            
            # Save the best model immediately
            if model_dir:
                model_save_path = model_dir / "best_model.pth"
                torch.save({
                    'epoch': best_epoch,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': best_val_loss,
                }, model_save_path)
                print(f"✓ New best model saved at epoch {best_epoch} with val loss: {best_val_loss:.4f}")

    # After training completes, you can also save the final model
    if model_dir:
        final_model_path = model_dir / "final_model.pth"
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
        }, final_model_path)
        print(f"Final model saved at: {final_model_path}")
    
    # # Load the best model back for return (optional)
    # if best_model_state is not None:
    #     model.load_state_dict(best_model_state)
    #     print(f"Loaded best model from epoch {best_epoch} with validation loss: {best_val_loss:.4f}")
    
    return model, best_val_loss