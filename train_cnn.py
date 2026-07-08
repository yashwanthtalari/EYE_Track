import os
import sys
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class GazeDataset(Dataset):
    def __init__(self, df, img_dir, img_size=(32, 32)):
        self.df = df
        self.img_dir = img_dir
        self.img_size = img_size
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, str(row["image"]))
        bgr = cv2.imread(img_path)
        if bgr is None:
            img = np.zeros((1, self.img_size[1], self.img_size[0]), dtype=np.float32)
        else:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            resized = cv2.resize(gray, self.img_size, interpolation=cv2.INTER_AREA)
            img = resized.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
            
        target = np.array([float(row["target_norm_x"]), float(row["target_norm_y"])], dtype=np.float32)
        return torch.tensor(img), torch.tensor(target)

class GazeCNN(nn.Module):
    def __init__(self):
        super(GazeCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x16 -> 8x8
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # 8x8 -> 4x4
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        
    def forward(self, x):
        features = self.cnn(x)
        features = features.view(features.size(0), -1)
        return self.fc(features)

def calculate_pixel_error(preds, targets, screen_w, screen_h):
    dx = (preds[:, 0] - targets[:, 0]) * screen_w
    dy = (preds[:, 1] - targets[:, 1]) * screen_h
    errs = np.sqrt(dx ** 2 + dy ** 2)
    return float(np.mean(errs)), float(np.median(errs))

def main():
    print("==================================================", flush=True)
    print("       VisionPoint: CNN + ANN Gaze Estimator       ", flush=True)
    print("==================================================", flush=True)

    data_dir = "dataset"
    labels_path = os.path.join(data_dir, "labels.csv")
    images_dir = os.path.join(data_dir, "images")
    
    if not os.path.exists(labels_path):
        print(f"Error: {labels_path} not found. Run collect_data.py first.")
        return
        
    df = pd.read_csv(labels_path)
    screen_w = int(df["screen_w"].median())
    screen_h = int(df["screen_h"].median())
    print(f"Loaded {len(df)} labeled images. Resolution: {screen_w}x{screen_h}", flush=True)
    
    # Train / Test split (20% held out)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Train samples: {len(train_df)} | Test samples: {len(test_df)}", flush=True)
    
    train_dataset = GazeDataset(train_df, images_dir, img_size=(32, 32))
    test_dataset = GazeDataset(test_df, images_dir, img_size=(32, 32))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Initialize model
    model = GazeCNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    print("\nTraining CNN + ANN model...", flush=True)
    
    epochs = 12
    best_test_loss = float('inf')
    best_model_weights = None
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            
        train_loss /= len(train_dataset)
        
        # Validation evaluation
        model.eval()
        test_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)
                all_preds.append(outputs.numpy())
                all_targets.append(targets.numpy())
                
        test_loss /= len(test_dataset)
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        mean_px, med_px = calculate_pixel_error(all_preds, all_targets, screen_w, screen_h)
        
        print(f"Epoch {epoch:2d}/{epochs} | Train Loss: {train_loss:.5f} | Test Loss: {test_loss:.5f} | "
              f"Pixel Error: mean={mean_px:.1f}px, median={med_px:.1f}px", flush=True)
              
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    # Load best weights
    model.load_state_dict(best_model_weights)
    
    # Save the model payload
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "gaze_cnn_model.pth")
    
    payload = {
        "model_state": best_model_weights,
        "img_size": (32, 32),
        "screen_w": screen_w,
        "screen_h": screen_h
    }
    torch.save(payload, model_path)
    
    print("\n==================================================", flush=True)
    print(f"Model saved successfully to: {model_path}", flush=True)
    print("Evaluation on Test Set (Best Model):", flush=True)
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            all_preds.append(outputs.numpy())
            all_targets.append(targets.numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    mean_px, med_px = calculate_pixel_error(all_preds, all_targets, screen_w, screen_h)
    
    def r2(yt, yp):
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        
    r2x = r2(all_targets[:, 0], all_preds[:, 0])
    r2y = r2(all_targets[:, 1], all_preds[:, 1])
    
    print(f"  * Mean Pixel Error  : {mean_px:.1f} px", flush=True)
    print(f"  * Median Pixel Error: {med_px:.1f} px", flush=True)
    print(f"  * X Coordinate R^2  : {r2x:.3f}", flush=True)
    print(f"  * Y Coordinate R^2  : {r2y:.3f}", flush=True)
    print("==================================================", flush=True)

if __name__ == "__main__":
    main()
