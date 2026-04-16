import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import math
import cv2
from scipy import ndimage
import statistics

# --- Make the camera input smaller ---
st.markdown("""
<style>
[data-testid="stCameraInput"] video {
    width: 300px !important;   /* Adjust width */
    height: auto !important;   /* Keep proper aspect ratio */
    border-radius: 10px;       /* Optional rounded corners */
}

[data-testid="stCameraInput"] button {
    transform: scale(0.8);     /* Make the capture button smaller */
}
</style>
""", unsafe_allow_html=True)

# ---------------- Load Model ----------------
model = tf.keras.models.load_model("model_new/fruit_mobilenetv2_corrected.keras")
CLASS_NAMES = ['Apple', 'Banana', 'Cherry', 'Mango', 'Orange', 'Papaya', 'Pineapple', 'Strawberry']

# ---------------- USDA API ----------------
API_KEY = "CefTFO6wbpUkVfpEPovZdegO8k3cOh19p4FIDZ2H"
USDA_API_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"

# ---------------- Improved Fruit Data ----------------
FRUIT_DENSITY = {
    "Apple": 0.65,      # Increased for better accuracy
    "Banana": 0.75,     # Increased
    "Cherry": 0.85,     # Adjusted
    "Mango": 0.82,      # Adjusted
    "Orange": 0.65,     # Increased
    "Papaya": 0.52,     # Adjusted
    "Pineapple": 0.85,  # Adjusted
    "Strawberry": 0.68  # Adjusted
}

FRUIT_SHAPE = {
    "Apple": "sphere",
    "Banana": "cylinder",
    "Cherry": "sphere",
    "Mango": "ellipsoid",
    "Orange": "sphere",
    "Papaya": "ellipsoid",
    "Pineapple": "cylinder",
    "Strawberry": "cone"
}

# Fruit-specific typical dimensions (width, height) in cm
TYPICAL_DIMENSIONS = {
    "Apple": (7.0, 7.0),      # diameter in cm
    "Banana": (3.0, 20.0),    # width, length in cm
    "Orange": (7.0, 7.0),     # diameter in cm
    "Mango": (6.0, 10.0),     # width, length in cm
    "Strawberry": (2.0, 3.0), # width, height in cm
    "Cherry": (2.0, 2.0),     # diameter in cm
    "Pineapple": (10.0, 20.0),# width, height in cm
    "Papaya": (8.0, 15.0),    # width, length in cm
}

# Reasonable weight ranges for validation (min, max) in grams
REASONABLE_WEIGHT_RANGES = {
    "Apple": (70, 250),
    "Banana": (80, 200),
    "Orange": (100, 300),
    "Mango": (150, 500),
    "Strawberry": (10, 30),
    "Cherry": (5, 15),
    "Pineapple": (500, 2000),
    "Papaya": (300, 1000),
}

# ---------------- Improved Scale Detection ----------------
def improve_scale_detection(image, mask, bbox, fruit_type):
    """Improved scale detection using known fruit dimensions"""
    x, y, w_pixels, h_pixels = bbox
    
    if fruit_type not in TYPICAL_DIMENSIONS:
        # Fallback to original method
        return detect_scale_reference(image)
    
    # Get typical dimensions for this fruit type
    typical_width_cm, typical_height_cm = TYPICAL_DIMENSIONS[fruit_type]
    
    # Calculate pixels per cm using both dimensions
    width_px_cm = w_pixels / typical_width_cm
    height_px_cm = h_pixels / typical_height_cm
    
    # Use weighted average (favor the more reliable measurement)
    if abs(w_pixels - h_pixels) / max(w_pixels, h_pixels) < 0.3:
        # Dimensions are similar, use simple average
        pixels_per_cm = (width_px_cm + height_px_cm) / 2
    else:
        # Dimensions differ significantly, use the more stable one
        pixels_per_cm = min(width_px_cm, height_px_cm)
    
    # Apply reasonable bounds (5-50 pixels/cm for typical phone cameras)
    pixels_per_cm = max(5.0, min(pixels_per_cm, 50.0))
    
    return pixels_per_cm

def detect_scale_reference(image):
    """Fallback scale detection method"""
    img_cv = np.array(image)
    if len(img_cv.shape) == 3 and img_cv.shape[2] == 3:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    
    height, width = img_cv.shape[:2]
    
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        object_area = cv2.contourArea(largest_contour)
        image_area = width * height
        area_ratio = object_area / image_area
        
        if area_ratio > 0.6:
            estimated_real_size_cm = 8.0
        elif area_ratio > 0.3:
            estimated_real_size_cm = 12.0
        else:
            estimated_real_size_cm = 20.0
        
        object_size_pixels = max(w, h)
        pixels_per_cm = object_size_pixels / estimated_real_size_cm
    else:
        diagonal_pixels = math.sqrt(width**2 + height**2)
        pixels_per_cm = diagonal_pixels / 15.0
    
    pixels_per_cm = max(10.0, min(pixels_per_cm, 100.0))
    return pixels_per_cm

# ---------------- Improved Volume Calculation ----------------
def calculate_accurate_volume(mask, fruit_type, pixels_per_cm):
    """More accurate volume calculations with correction factors"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0
    
    contour = max(contours, key=cv2.contourArea)
    x, y, w_pixels, h_pixels = cv2.boundingRect(contour)
    
    # Convert to cm
    width_cm = w_pixels / pixels_per_cm
    height_cm = h_pixels / pixels_per_cm
    
    # More realistic volume calculations with correction factors
    shape = FRUIT_SHAPE[fruit_type]
    
    if shape == "sphere":
        # Use average diameter for spheres with correction
        avg_diameter = (width_cm + height_cm) / 2
        radius = avg_diameter / 2
        volume = (4/3) * math.pi * (radius ** 3) * 0.8  # 0.8 correction factor
        
    elif shape == "cylinder":
        if fruit_type == "Banana":
            # Banana is curved - use elliptical cross-section with smaller factors
            radius_minor = (min(width_cm, height_cm) / 2) * 0.4  # Reduced from 0.6
            radius_major = (max(width_cm, height_cm) / 2) * 0.6  # Reduced from 0.8
            length = max(width_cm, height_cm)
            volume = math.pi * radius_minor * radius_major * length * 0.7  # Curvature correction
        else:  # Pineapple
            radius = (width_cm / 2) * 0.5  # Reduced from 0.7
            height = height_cm
            volume = math.pi * (radius ** 2) * height * 0.9  # Tapering correction
            
    elif shape == "ellipsoid":
        if fruit_type == "Mango":
            a = width_cm / 2 * 0.8   # Correction factors
            b = height_cm / 2 * 0.8
            c = (a + b) / 3
        else:  # Papaya
            a = width_cm / 2 * 0.7
            b = height_cm / 2 * 0.7
            c = (a + b) / 2
        volume = (4/3) * math.pi * a * b * c
        
    elif shape == "cone":
        # Strawberry - cone shape with corrections
        radius = width_cm / 2 * 0.7
        height = height_cm * 0.8
        volume = (1/3) * math.pi * (radius ** 2) * height
        
    else:
        # Conservative fallback
        volume = width_cm * height_cm * min(width_cm, height_cm) * 0.5
    
    return max(volume, 0.1)  # Ensure minimum volume

# ---------------- Improved Weight Estimation ----------------
def estimate_weight_from_silhouette(mask, fruit_type, pixels_per_cm):
    """Improved weight estimation with validation"""
    volume = calculate_accurate_volume(mask, fruit_type, pixels_per_cm)
    density = FRUIT_DENSITY[fruit_type]
    
    weight = density * volume
    
    # Apply fruit-specific weight validation
    if fruit_type in REASONABLE_WEIGHT_RANGES:
        min_weight, max_weight = REASONABLE_WEIGHT_RANGES[fruit_type]
        
        # If weight is outside reasonable range, use geometric mean of range
        if weight < min_weight or weight > max_weight:
            # Calculate adjustment factor
            if weight > max_weight:
                adjustment_factor = max_weight / weight
            else:
                adjustment_factor = min_weight / weight
            
            # Apply gradual adjustment
            weight = weight * (0.3 + 0.7 * adjustment_factor)
    
    return max(weight, 1)  # Ensure minimum weight

def validate_weight_estimation(weight, fruit_type, volume, width, height):
    """Validate weight using statistical reasoning"""
    expected_density = FRUIT_DENSITY[fruit_type]
    calculated_density = weight / volume if volume > 0 else 0
    
    # If density is unreasonable, adjust weight
    if calculated_density < expected_density * 0.3 or calculated_density > expected_density * 3.0:
        weight = expected_density * volume
    
    # Ensure weight is physically plausible
    max_reasonable_weight = volume * 2.0
    min_reasonable_weight = volume * 0.3
    
    weight = max(min(weight, max_reasonable_weight), min_reasonable_weight)
    
    return weight

# ---------------- Segmentation Functions (Unchanged) ----------------
def segment_fruit_advanced(image):
    """Advanced fruit segmentation using multiple techniques"""
    # Convert PIL to OpenCV
    img_cv = np.array(image)
    if len(img_cv.shape) == 3 and img_cv.shape[2] == 3:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    
    original_img = img_cv.copy()
    
    # Method 1: GrabCut for precise segmentation
    mask_grabcut = grabcut_segmentation(img_cv)
    
    # Method 2: Color-based segmentation
    mask_color = color_based_segmentation(img_cv)
    
    # Method 3: Edge-based segmentation
    mask_edges = edge_based_segmentation(img_cv)
    
    # Combine masks
    combined_mask = combine_masks(mask_grabcut, mask_color, mask_edges)
    
    # Refine mask
    refined_mask = refine_mask(combined_mask)
    
    # Extract largest connected component
    final_mask = extract_largest_component(refined_mask)
    
    # Get bounding box and contours
    bbox, contours = get_bounding_box_and_contours(final_mask)
    
    return final_mask, bbox, contours, original_img

def grabcut_segmentation(img):
    """Use GrabCut algorithm for precise segmentation"""
    mask = np.zeros(img.shape[:2], np.uint8)
    
    # Initialize background and foreground models
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    # Define rectangle around the center (assuming fruit is roughly centered)
    h, w = img.shape[:2]
    rect = (w//8, h//8, w*3//4, h*3//4)
    
    try:
        cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    except:
        # Fallback: use entire image
        mask = np.ones(img.shape[:2], np.uint8) * 255
        return mask
    
    # Create mask where sure and likely foreground
    mask_grabcut = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8') * 255
    
    return mask_grabcut

def color_based_segmentation(img):
    """Color-based segmentation in multiple color spaces"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    masks = []
    
    # HSV color ranges for fruits
    color_ranges = [
        # Red fruits
        ([0, 50, 50], [10, 255, 255]),
        ([170, 50, 50], [180, 255, 255]),
        # Yellow/Orange fruits
        ([15, 50, 50], [35, 255, 255]),
        # Green fruits
        ([36, 50, 50], [85, 255, 255]),
    ]
    
    for lower, upper in color_ranges:
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        masks.append(mask)
    
    # Combine all color masks
    combined_color_mask = np.zeros(img.shape[:2], np.uint8)
    for mask in masks:
        combined_color_mask = cv2.bitwise_or(combined_color_mask, mask)
    
    return combined_color_mask

def edge_based_segmentation(img):
    """Edge-based segmentation to capture boundaries"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while keeping edges sharp
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Detect edges
    edges = cv2.Canny(blurred, 30, 150)
    
    # Dilate edges to connect gaps
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    
    # Fill enclosed regions
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(edges)
    cv2.drawContours(mask, contours, -1, 255, -1)
    
    return mask

def combine_masks(mask1, mask2, mask3):
    """Combine multiple masks using logical OR"""
    combined = np.zeros_like(mask1)
    combined = cv2.bitwise_or(combined, mask1)
    combined = cv2.bitwise_or(combined, mask2)
    combined = cv2.bitwise_or(combined, mask3)
    return combined

def refine_mask(mask):
    """Refine mask using morphological operations"""
    kernel = np.ones((5, 5), np.uint8)
    
    # Close small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Fill holes
    mask = ndimage.binary_fill_holes(mask).astype(np.uint8) * 255
    
    return mask

def extract_largest_component(mask):
    """Extract the largest connected component"""
    # Label connected components
    labeled_mask, num_features = ndimage.label(mask)
    
    if num_features == 0:
        return mask
    
    # Find largest component
    sizes = ndimage.sum(mask, labeled_mask, range(1, num_features + 1))
    if len(sizes) == 0:
        return mask
    
    largest_component = np.argmax(sizes) + 1
    final_mask = (labeled_mask == largest_component).astype(np.uint8) * 255
    
    return final_mask

def get_bounding_box_and_contours(mask):
    """Get bounding box and contours from mask"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return (0, 0, mask.shape[1], mask.shape[0]), []
    
    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Add padding
    padding = 15
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(mask.shape[1] - x, w + 2 * padding)
    h = min(mask.shape[0] - y, h + 2 * padding)
    
    return (x, y, w, h), [largest_contour]

def extract_fruit_from_mask(image, mask, bbox):
    """Extract fruit region using the segmentation mask"""
    x, y, w, h = bbox
    
    # Create masked image
    masked_img = image.copy()
    if len(masked_img.shape) == 3:
        mask_3d = np.stack([mask] * 3, axis=-1)
        masked_img[mask_3d == 0] = 0
    
    # Crop to bounding box
    fruit_region = masked_img[y:y+h, x:x+w]
    fruit_mask = mask[y:y+h, x:x+w]
    
    return fruit_region, fruit_mask

# ---------------- Other Functions ----------------
def fetch_calories(fruit_name):
    try:
        params = {
            'api_key': API_KEY,
            'query': f"{fruit_name}, raw",
            'pageSize': 1
        }
        response = requests.get(USDA_API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        if not data['foods']:
            return None
        food_item = data['foods'][0]
        nutrients = food_item.get('foodNutrients', [])
        for nutrient in nutrients:
            if nutrient.get('nutrientNumber') == '208':  # Calories
                return nutrient.get('value')
        return None
    except:
        return None

def classify_image(image):
    # Advanced segmentation first
    mask, bbox, contours, img_cv = segment_fruit_advanced(image)
    
    # Extract fruit region
    fruit_region, fruit_mask = extract_fruit_from_mask(img_cv, mask, bbox)
    
    # Convert to PIL for classification
    fruit_pil = Image.fromarray(cv2.cvtColor(fruit_region, cv2.COLOR_BGR2RGB))
    
    # Preprocess for classification
    img_classify = fruit_pil.resize((224, 224))
    img_array = np.array(img_classify) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array, verbose=0)
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx]
    
    # CONFIDENCE THRESHOLD - if confidence is too low, return "No Fruit Detected"
    CONFIDENCE_THRESHOLD = 0.7
    
    if confidence < CONFIDENCE_THRESHOLD:
        fruit_type = "No Fruit Detected"
        return fruit_type, confidence, {}, None, 0, fruit_pil, mask, bbox, contours, 0
    
    fruit_type = CLASS_NAMES[class_idx]
    
    # IMPROVED: Get scale AFTER classification using fruit-specific dimensions
    pixels_per_cm = improve_scale_detection(image, mask, bbox, fruit_type)
    
    # Top 3 predictions
    top_3_idx = np.argsort(predictions[0])[-3:][::-1]
    top_3 = {CLASS_NAMES[idx]: predictions[0][idx] for idx in top_3_idx}

    # Estimate weight using improved method
    calories = fetch_calories(fruit_type)
    weight = estimate_weight_from_silhouette(mask, fruit_type, pixels_per_cm)

    return fruit_type, confidence, top_3, calories, weight, fruit_pil, mask, bbox, contours, pixels_per_cm

def display_results(original_image, fruit_type, confidence, top_3, calories, weight, fruit_image, mask, bbox, contours, pixels_per_cm):
    """Display analysis results in a structured format"""
    
    # Display results in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(original_image, caption="Original Image", use_container_width=True)
    
    with col2:
        # Create visualization of segmentation
        img_cv = np.array(original_image)
        if len(img_cv.shape) == 3:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        
        # Draw contours on original image
        visualization = img_cv.copy()
        if contours:
            cv2.drawContours(visualization, contours, -1, (0, 255, 0), 3)
        
        # Draw bounding box
        x, y, w, h = bbox
        cv2.rectangle(visualization, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        visualization_rgb = cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB)
        st.image(visualization_rgb, caption="Segmentation Result", use_container_width=True)
    
    with col3:
        st.image(fruit_image, caption="Extracted Fruit", use_container_width=True)
    
    # Display mask and technical info
    col4, col5 = st.columns(2)
    
    with col4:
        st.image(mask, caption="Segmentation Mask", use_container_width=True, clamp=True)
    
    with col5:
        # Create masked image
        img_cv = np.array(original_image)
        if len(img_cv.shape) == 3:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        masked_original = img_cv.copy()
        mask_3d = np.stack([mask] * 3, axis=-1) if len(masked_original.shape) == 3 else mask
        masked_original[mask_3d == 0] = 0
        masked_original_rgb = cv2.cvtColor(masked_original, cv2.COLOR_BGR2RGB)
        st.image(masked_original_rgb, caption="Masked Image", use_container_width=True)

    # Results
    st.subheader("🎯 Prediction Results")
    col6, col7, col8 = st.columns(3)
    
    with col6:
        st.metric("Fruit Type", fruit_type)
        st.metric("Estimated Weight", f"{weight:.1f} g")
        st.metric("Confidence", f"{confidence:.2%}")
    
    with col7:
        if calories:
            estimated_calories = (weight / 100) * calories
            st.metric("Calories per 100g", f"{calories} kcal")
            st.metric("Estimated Calories", f"{estimated_calories:.1f} kcal")
            st.caption(f"Based on {weight:.1f}g fruit × {calories}kcal/100g")
        else:
            st.metric("Calories per 100g", "Not available")
            st.metric("Estimated Calories", "Not available")

    # Nutritional Information Section
    st.subheader("🍎 Nutritional Information")
    if calories:
        col9, col10, col11 = st.columns(3)
        
        with col9:
            st.metric("Estimated Weight", f"{weight:.1f} g")
            
        with col10:
            st.metric("Calories per 100g", f"{calories} kcal")
            
        with col11:
            st.metric("Total Calories", f"{(weight/100 * calories):.1f} kcal")

    # Show scale information for debugging
    with st.expander("📏 Technical Details"):
        #st.write(f"**Scale Detection**: {pixels_per_cm:.1f} pixels/cm")
        st.write(f"**Fruit Type**: {fruit_type}")
        st.write(f"**Confidence**: {confidence:.2%}")
        #st.write("**Top Predictions**:")
        #for fruit, conf in top_3.items():
            #st.write(f"- {fruit}: {conf:.2%}")

# ---------------- Streamlit App ----------------
st.title("🍎 Advanced Fruit Weight & Calories Analysis")
st.write("Upload an image or use your camera for precise fruit segmentation, 3D shape analysis, and accurate weight estimation.")

# Calibration section
st.sidebar.title("⚙️ Calibration")
known_weight = st.sidebar.number_input("If you know the actual weight (g):", min_value=0.0, value=0.0, step=1.0)
calibration_note = st.sidebar.empty()

# Input method selection
input_method = st.radio("Choose input method:", ["File Upload", "Camera"])

if input_method == "File Upload":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        
        with st.spinner("Performing advanced segmentation and analysis..."):
            fruit_type, confidence, top_3, calories, weight, fruit_image, mask, bbox, contours, pixels_per_cm = classify_image(image)
        
        # Check if no fruit detected
        if fruit_type == "No Fruit Detected":
            st.error("❌ No fruit detected in the image. Please try again with a clear image of a fruit.")
            st.image(image, caption="Original Image", use_container_width=True)
            st.write(f"Highest confidence: {confidence:.2%} (below threshold)")
        else:
            # Show calibration info if known weight provided
            if known_weight > 0:
                calibration_factor = known_weight / weight
                calibration_note.info(f"Calibration factor: {calibration_factor:.2f} (Actual/Estimated)")
                if abs(calibration_factor - 1.0) > 0.3:
                    calibration_note.warning("Large calibration needed. Check fruit positioning and lighting.")
            
            display_results(image, fruit_type, confidence, top_3, calories, weight, fruit_image, mask, bbox, contours, pixels_per_cm)

else:  # Camera input
    st.write("### 📸 Real-time Camera Analysis")
    st.write("Position the fruit clearly in the camera frame and capture an image for analysis.")
    
    # Camera input
    camera_image = st.camera_input("Take a picture of the fruit")
    
    if camera_image:
        image = Image.open(camera_image)
        
        with st.spinner("Performing real-time analysis..."):
            fruit_type, confidence, top_3, calories, weight, fruit_image, mask, bbox, contours, pixels_per_cm = classify_image(image)
        
        # Check if no fruit detected
        if fruit_type == "No Fruit Detected":
            st.error("❌ No fruit detected in the image. Please try again with a clear image of a fruit.")
            st.image(image, caption="Original Image", use_container_width=True)
            st.write(f"Highest confidence: {confidence:.2%} (below threshold)")
        else:
            # Show calibration info if known weight provided
            if known_weight > 0:
                calibration_factor = known_weight / weight
                calibration_note.info(f"Calibration factor: {calibration_factor:.2f} (Actual/Estimated)")
                if abs(calibration_factor - 1.0) > 0.3:
                    calibration_note.warning("Large calibration needed. Check fruit positioning and lighting.")
            
            display_results(image, fruit_type, confidence, top_3, calories, weight, fruit_image, mask, bbox, contours, pixels_per_cm)

# Add some tips for better results
st.sidebar.title("💡 Tips for Better Results")
st.sidebar.write("""
- **Good Lighting**: Ensure the fruit is well-lit without harsh shadows
- **Plain Background**: Use a contrasting, plain background for better segmentation
- **Single Fruit**: Analyze one fruit at a time for accurate results
- **Clear View**: Make sure the fruit is fully visible and not obscured
- **Camera Distance**: Keep the camera about 30-50cm from the fruit
- **Focus**: Ensure the fruit is in focus for better feature extraction
- **Known Weight**: Use the calibration feature if you know the actual weight
""")

st.sidebar.title("ℹ️ About")
st.sidebar.write("""
This app uses advanced computer vision techniques:
- **Improved Scale Detection**: Uses fruit-specific typical dimensions
- **Advanced Segmentation**: Combines multiple algorithms for precise fruit extraction
- **3D Shape Analysis**: Estimates volume based on fruit geometry with correction factors
- **Machine Learning**: Classifies fruits using a trained MobileNetV2 model
- **Nutritional Data**: Fetches calorie information from USDA database

""")

