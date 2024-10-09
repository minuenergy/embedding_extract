import os
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from tqdm import tqdm

def load_matched_images(satellite_dir, drone_dir):
    matched_images = []

    # Iterate through satellite folders
    for folder_name in os.listdir(satellite_dir):
        satellite_folder_path = os.path.join(satellite_dir, folder_name)
        drone_folder_path = os.path.join(drone_dir, folder_name)

        # Check if both satellite and drone folders exist
        if os.path.isdir(satellite_folder_path) and os.path.isdir(drone_folder_path):
            # Find the satellite image
            satellite_images = [f for f in os.listdir(satellite_folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            if satellite_images:
                satellite_image_path = os.path.join(satellite_folder_path, satellite_images[0])
                
                # Find all drone images
                drone_images = [f for f in os.listdir(drone_folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
                drone_image_paths = [os.path.join(drone_folder_path, f) for f in drone_images]

                # If we have both satellite and drone images, add them to the list
                if drone_image_paths:
                    matched_images.append({
                        'folder': folder_name,
                        'satellite': satellite_image_path,
                        'drones': drone_image_paths
                    })

    return matched_images

def extract_embed(model, device, prompt, image):
    inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True)
    inputs.to(device)
    with torch.no_grad():
        with torch.autocast(device):
            outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

    text_features = outputs['text_model_output']['pooler_output']
    image_features = outputs['vision_model_output']['pooler_output']
    combined_features = torch.cat([image_features, text_features.mean(dim=0, keepdim=True)], dim=1)
    
    return text_features, image_features, combined_features



def save_embeddings(base_path, folder_name, image_name, embeddings_dict):
    for embed_type, embed_data in embeddings_dict.items():
        embed_folder = os.path.join(base_path, embed_type, folder_name)
        os.makedirs(embed_folder, exist_ok=True)
        
        file_name = os.path.splitext(image_name)[0] + '.npy'
        file_path = os.path.join(embed_folder, file_name)
        
        np.save(file_path, embed_data.cpu().numpy())

def process_and_save_embeddings(matched_image_sets, model, processor, device, prompt, base_save_path):
    for image_set in tqdm(matched_image_sets):
        folder_name = image_set['folder']
        print(f"Processing folder: {folder_name}")
        
        # Process satellite image
        satellite_img = Image.open(image_set['satellite'])
        sat_text_features, sat_image_features, sat_combined_features = extract_embed(model, device, prompt, satellite_img)
        
        # Save satellite embeddings
        sat_embeddings = {
            'text_features': sat_text_features,
            'image_features': sat_image_features,
            'combined_features': sat_combined_features
        }
        satellite_name = os.path.basename(image_set['satellite'])
        save_embeddings(os.path.join(base_save_path, 'satellite'), folder_name, satellite_name, sat_embeddings)
        
        # Process drone images
        for drone_img_path in image_set['drones']:
            drone_img = Image.open(drone_img_path)
            drone_text_features, drone_image_features, drone_combined_features = extract_embed(model, device, prompt, drone_img)
            
            # Save drone embeddings
            drone_embeddings = {
                'text_features': drone_text_features,
                'image_features': drone_image_features,
                'combined_features': drone_combined_features
            }
            drone_name = os.path.basename(drone_img_path)
            save_embeddings(os.path.join(base_save_path, 'drone'), folder_name, drone_name, drone_embeddings)

if __name__ == "__main__":
    satellite_directory = "../FSRA/University-Release/train/satellite"
    drone_directory = "../FSRA/University-Release/train/drone"
    base_save_path = 'embeddings'
    prompt = "building, road"

    device = "cuda"
    torch_dtype = torch.float16
    model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32",
        attn_implementation="flash_attention_2",
        device_map=device,
        torch_dtype=torch_dtype,
    )
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    matched_image_sets = load_matched_images(satellite_directory, drone_directory)
    process_and_save_embeddings(matched_image_sets, model, processor, device, prompt, base_save_path)

    print("Embeddings saved successfully!")