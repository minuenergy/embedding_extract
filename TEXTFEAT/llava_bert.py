import os
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel, BertTokenizer, BertModel
import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams


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

def generate_caption(llm, image_path, question):
    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    sampling_params = SamplingParams(temperature=0.01, max_tokens=512)

    image = Image.open(image_path)
    inputs = {
        "prompt": prompt,
        "multi_modal_data": {
            "image": image
        },
    }
    outputs = llm.generate(inputs, sampling_params=sampling_params)
    return outputs[0].outputs[0].text

def save_captions(captions, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for image_path, caption in captions:
            f.write(f"{image_path},{caption}\n")

def create_bert_embedding(text, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

def save_embeddings(base_path, folder_name, image_name, embeddings):
    embed_folder = os.path.join(base_path, 'text_features', folder_name)
    os.makedirs(embed_folder, exist_ok=True)
    
    file_name = os.path.splitext(image_name)[0] + '.npy'
    file_path = os.path.join(embed_folder, file_name)
    
    np.save(file_path, embeddings.cpu().numpy())

def process_images_and_generate_captions(matched_image_sets, llm, question, output_file):
    captions = []
    for image_set in tqdm(matched_image_sets, desc="Generating captions"):
        # Process satellite image
        satellite_caption = generate_caption(llm, image_set['satellite'], question)
        captions.append((image_set['satellite'], satellite_caption))
        
        # Process drone images
        for drone_img_path in image_set['drones']:
            drone_caption = generate_caption(llm, drone_img_path, question)
            captions.append((drone_img_path, drone_caption))
    
    save_captions(captions, output_file)
    return captions

def create_and_save_bert_embeddings(captions, tokenizer, model, device, base_save_path):
    for image_path, caption in tqdm(captions, desc="Creating BERT embeddings"):
        embedding = create_bert_embedding(caption, tokenizer, model, device)
        
        # Determine if it's a satellite or drone image
        if 'satellite' in image_path:
            base_path = os.path.join(base_save_path, 'satellite')
        else:
            base_path = os.path.join(base_save_path, 'drone')
        
        # Extract folder name and image name
        parts = image_path.split(os.sep)
        folder_name = parts[-2]
        image_name = parts[-1]
        
        save_embeddings(base_path, folder_name, image_name, embedding)

if __name__ == "__main__":
    satellite_directory = "../FSRA/University-Release/train/satellite"
    drone_directory = "../FSRA/University-Release/train/drone"
    base_save_path = 'llava_bert_embeddings'
    caption_file = 'university1652_llava15_caption.txt'
    question = "Describe this image with main building or landmark"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load LLaVA model
    llm = LLM(model="llava-hf/llava-1.5-7b-hf", max_model_len=1024)

    # Load BERT model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

    matched_image_sets = load_matched_images(satellite_directory, drone_directory)
    
    # Generate and save captions
    captions = process_images_and_generate_captions(matched_image_sets, llm, question, caption_file)
    print(f"Captions saved to {caption_file}")

    # Create and save BERT embeddings
    create_and_save_bert_embeddings(captions, tokenizer, bert_model, device, base_save_path)
    print("BERT embeddings saved successfully!")
