import tensorflow as tf
import json

print("Loading model config from JSON...")
with open('models/model.json', 'r') as f:
    model_json = json.load(f)

print("First 5 layers:")
for layer in model_json['config']['layers'][:5]:
    name = layer['config'].get('name', 'N/A')
    cls = layer['class_name']
    print(f"  {cls}: {name}")

# Try loading with custom_objects
print("\nTrying to load weights...")
try:
    model = tf.keras.models.model_from_json(json.dumps(model_json['config']))
    model.load_weights('models/weights.h5')
    print("Success! Model loaded from JSON + weights")
    
    # Save as new format
    saved_path = 'models/facenet_keras_converted.h5'
    model.save(saved_path)
    print(f"Saved as: {saved_path}")
except Exception as e:
    print(f"Failed: {e}")