# Chest X-Ray Pneumonia Classification Project

## Project Overview

This project aims to develop a high-accuracy deep learning model for automatically distinguishing between normal chest X-ray images and pneumonia chest X-ray images. The project goal is to achieve a classification accuracy of over 90%, while focusing on model performance on an imbalanced dataset.

## Dataset

The project uses a widely applied chest X-ray pneumonia dataset from Kaggle:
- Data source: Paul Mooney's "Chest X-Ray Pneumonia" Kaggle dataset
- Classes: NORMAL and PNEUMONIA (two classes)
- Dataset structure: Divided into training, validation, and test sets

## Introduction to Transfer Learning Models

### ResNet (Residual Network)

ResNet solves the training difficulty of deep neural networks by introducing "skip connections". These connections allow information to bypass certain layers, effectively alleviating the vanishing gradient problem.

- **ResNet-50**: Contains 50 layers composed of multiple residual blocks, offering a balanced choice between performance and computational cost.
- **ResNet-101**: With 101 layers, deeper than ResNet-50, capable of capturing more complex features but requires more computational resources.

### DenseNet (Densely Connected Network)

DenseNet implements interconnection of all layers within each dense block, meaning each layer receives input from all previous layers. This architecture promotes feature reuse, mitigates the vanishing gradient problem, and significantly reduces the number of parameters.

- **DenseNet-121**: Has 121 layers, efficiently utilizes features, contains fewer parameters than ResNet of equivalent depth, and has good resistance to overfitting.

## Optimization Strategies

### 1. Data Augmentation Techniques

To enhance model generalization ability and mitigate limited data issues, various data augmentation techniques were implemented:

```python
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

### 2. Handling Class Imbalance

To address class imbalance in the dataset, a weighted loss function was implemented:

```python
def calculate_class_weights(dataset):
    labels = np.array(dataset.labels)
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    
    # Enhanced inverse frequency weighting
    weights = total_samples / (len(class_counts) * class_counts)
    
    # Further boost the minority class weight
    minority_idx = np.argmin(class_counts)
    weights[minority_idx] *= 1.3
    
    # Normalize weights
    weights = weights / weights.sum()
    
    return torch.FloatTensor(weights)
```

### 3. Feature Extraction and Fine-Tuning Strategy

A flexible layer freezing strategy was implemented for different transfer learning scenarios:

```python
if freeze_layers:
    # For ResNet models
    if hasattr(model, 'fc'):
        # Freeze early layers but keep the last block trainable
        for name, param in model.named_parameters():
            if 'layer4' not in name and 'fc' not in name:  # Only train layer4 and fc
                param.requires_grad = False
    # For DenseNet models
    elif hasattr(model, 'classifier'):
        # Freeze all except the last dense block and classifier
        for name, param in model.named_parameters():
            if 'denseblock4' not in name and 'classifier' not in name:
                param.requires_grad = False
```

### 4. Custom Classification Head Design

To improve model performance, a more complex classification head was added to replace a simple fully connected layer:

```python
# Custom classification head for ResNet models
model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, num_classes)
)
```

### 5. Advanced Training Techniques

#### Cosine Annealing Learning Rate Scheduler

```python
# Learning rate scheduler - Cosine Annealing
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
```

#### Gradient Clipping

```python
# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

#### Mixed Precision Training

```python
# Use mixed precision training for acceleration
with torch.cuda.amp.autocast():
    outputs = model(inputs)
    loss = criterion(outputs, labels)

# Scale loss and call backward()
scaler.scale(loss).backward()
```

### 6. Test-Time Augmentation (TTA)

Improving inference accuracy by applying multiple transformations to test images and averaging the prediction results:

```python
def test_time_augmentation(model, image, num_augmentations=10):
    model.eval()
    
    # Define TTA transformations
    tta_transforms = []
    
    # Original prediction
    tta_transforms.append(lambda x: x)
    
    # Horizontal flip
    tta_transforms.append(lambda x: torch.flip(x, dims=[3]))
    
    # Vertical flip
    tta_transforms.append(lambda x: torch.flip(x, dims=[2]))
    
    # 90-degree rotation
    tta_transforms.append(lambda x: torch.rot90(x, k=1, dims=[2, 3]))
    
    # Apply transformations and get predictions
    with torch.no_grad():
        predictions = []
        
        for transform in tta_transforms[:num_augmentations]:
            augmented_image = transform(image)
            output = model(augmented_image)
            predictions.append(output)
        
        # Average predictions
        avg_prediction = torch.mean(torch.stack(predictions), dim=0)
    
    return avg_prediction
```

### 7. Model Ensemble

When a single model fails to achieve the target accuracy, model ensembling is used to further improve performance:

```python
# Get predictions from two models
outputs1 = model(inputs)
outputs2 = model2(inputs)

# Average the prediction results (simple ensemble)
ensemble_outputs = (outputs1 + outputs2) / 2
```

## Evaluation Metrics

To comprehensively evaluate model performance, multiple evaluation metrics are used:

- Accuracy
- Balanced Accuracy
- Precision
- Recall
- F1 Score
- Average Precision
- Confusion Matrix

## Experimental Results and Visualization

The project generates the following visualizations to help analyze model performance:

1. Training history curves: Show changes in accuracy and loss during the training process
2. Confusion matrix: Visually display classification results
3. Model comparison table: Compare performance across different model architectures

## Future Improvement Directions

If model performance still does not meet requirements, consider the following optimization directions:

1. Try more test-time augmentation transformations
2. Further adjust class weights
3. Use larger models, such as the EfficientNet series
4. Implement more aggressive data augmentation strategies
5. Explore attention mechanisms
6. Use more complex ensemble methods, such as stacking

## Conclusion

This project demonstrates how to leverage transfer learning and various optimization strategies to improve the accuracy of medical image classification tasks. By combining pre-trained models, data augmentation, class balancing, test-time augmentation, and model ensembling techniques, we can develop high-performance pneumonia detection systems to provide auxiliary support for clinical diagnosis. 