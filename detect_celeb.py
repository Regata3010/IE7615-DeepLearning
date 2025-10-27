import sys
from ultralytics import YOLO
from pathlib import Path
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def get_celebrity_mapping():
    """Get the mapping from class index to celebrity ID from CSV"""
    # Read your CSV to get the actual celebrity IDs
    df = pd.read_csv('cs1.csv')
    celebrity_ids = sorted(df['Celebrity Choice/ID'].unique())
    
    # Create mapping: class index (0-46) -> celebrity ID
    idx_to_celeb = {idx: int(celeb_id) for idx, celeb_id in enumerate(celebrity_ids)}
    
    print(f"Loaded {len(celebrity_ids)} celebrity mappings")
    return idx_to_celeb

def detect_and_identify(image_path, visualize=True):
    """Detect faces and show actual celebrity IDs"""
    
    if not Path(image_path).exists():
        print(f"Error: {image_path} not found")
        return
    
    # Get celebrity ID mapping
    idx_to_celeb = get_celebrity_mapping()
    
    # Load YOLO model
    print("Loading YOLO model...")
    model = YOLO('/Users/Deep Learning Core/detection_dataset_47/best.pt')
    
    # Run detection
    results = model.predict(source=image_path, conf=0.25, verbose=False)
    
    # Parse results with actual celebrity IDs
    detections = []
    celebrity_10173_count = 0
    
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().item() * 100
                class_idx = int(box.cls[0].cpu().item())
                
                # Map class index to actual celebrity ID
                celebrity_id = idx_to_celeb.get(class_idx, f"Unknown_{class_idx}")
                
                detections.append({
                    'celebrity_id': celebrity_id,
                    'confidence': conf,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'class_idx': class_idx
                })
                
                if celebrity_id == 10173:
                    celebrity_10173_count += 1
    
    # Print results
    print("\n" + "="*60)
    print("CELEBRITY DETECTION RESULTS")
    print("="*60)
    print(f"Image: {image_path}")
    print(f"Total faces detected: {len(detections)}")
    
    if celebrity_10173_count > 0:
        print(f"\n⭐⭐⭐ CELEBRITY 10173 FOUND {celebrity_10173_count} TIME(S)! ⭐⭐⭐")
    
    print("\n📊 Detected Celebrities:")
    for i, det in enumerate(detections, 1):
        star = " ⭐" if det['celebrity_id'] == 10173 else ""
        print(f"  Face {i}: Celebrity ID {det['celebrity_id']} ({det['confidence']:.1f}% confidence){star}")
        print(f"           Location: [{det['bbox'][0]}, {det['bbox'][1]}]")
    
    # Visualize if requested
    if visualize and len(detections) > 0:
        visualize_detections(image_path, detections)
    
    print("="*60)
    return detections

def visualize_detections(image_path, detections):
    """Show image with celebrity IDs labeled on each face"""
    
    # Load image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img_rgb)
    
    # Draw boxes with celebrity IDs
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        celebrity_id = det['celebrity_id']
        conf = det['confidence']
        
        # Special styling for celebrity 10173
        if celebrity_id == 10173:
            color = 'green'
            linewidth = 3
            fontsize = 12
        else:
            color = 'red'
            linewidth = 2
            fontsize = 10
        
        # Draw rectangle
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                linewidth=linewidth, edgecolor=color,
                                facecolor='none')
        ax.add_patch(rect)
        
        # Add celebrity ID label
        label = f"ID: {celebrity_id}\n{conf:.0f}%"
        ax.text(x1, y1-5, label, color='white', fontsize=fontsize,
               fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor=color, 
                        alpha=0.8, edgecolor='black'))
    
    # ax.set_title(f'Detected {len(detections)} Celebrities', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Save and show
    output_name = 'celebrity_detection_result.png'
    plt.savefig(output_name, dpi=150, bbox_inches='tight')
    print(f"\n📸 Visualization saved as '{output_name}'")
    plt.show()

def main():
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Default test image
        image_path = "add image path here"
        if not Path(image_path).exists():
            print("Usage: python detect_celebrities.py <image_path>")
            return
    
    detect_and_identify(image_path)

if __name__ == "__main__":
    main()