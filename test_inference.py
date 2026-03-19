import requests
import sys
import os

def test_api(audio_file_path):
    """
    A helper script that demonstrates how to send an audio file to the Flask backend.
    In the real world, your Flutter mobile app would do this exact process using Dart's http package!
    
    Args:
        audio_file_path (str): The local path to the test audio file (.wav, .mp3)
    """
    
    # URL of our local Flask server
    API_URL = "http://localhost:5000/detect_voice"
    
    # 1. Validation
    if not os.path.exists(audio_file_path):
        print(f"Error: The test file '{audio_file_path}' does not exist.")
        return

    print(f"Connecting to Vacha-Shield API at {API_URL}...")
    print(f"Uploading and analyzing file: {audio_file_path} \n")

    try:
        # 2. Open the file in binary mode
        with open(audio_file_path, 'rb') as fp:
            # We bundle it in a dictionary dictating the form-data fields. 
            # 'file' is the key our Flask API expects.
            files = {'file': fp}
            
            # Send HTTP POST request
            response = requests.post(API_URL, files=files)

        # 3. Analyze the response
        if response.status_code == 200:
            result = response.json()
            print("=== API Response (SUCCESS) ===")
            print(f"Synthetic Prob Score: {result['synthetic_probability'] * 100:.2f}%")
            print(f"Human Prob Score:     {result['human_probability'] * 100:.2f}%")
            print(f"Decision Threshold:   {result.get('threshold', 0.5) * 100:.2f}%")
            print(f"Final Verdict:        {'AI/Deepfake' if result.get('alert') else 'Human'}")
            
            # Highlight with a simple text-based warning if it's AI
            if result.get('alert'):
                print("\n[!] WARNING: POTENTIAL DEEPFAKE DETECTED [!]")
            else:
                print("\n[*] SAFE: Human Voice Detected [*]")
                
        else:
            print(f"=== API Response (ERROR {response.status_code}) ===")
            print(response.text)

    except requests.exceptions.ConnectionError:
        print("Error connecting to server. Is the Flask API running? Try running 'python app.py' in a separate terminal.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    # If no file is provided via command line, generate a dummy file to test
    if len(sys.argv) < 2:
        print("Usage: python test_inference.py <path_to_audio_file>")
        print("Generating a quick dummy '.wav' file using Librosa to test anyway...\n")
        
        # Create a dummy 1-second audio file using numpy and soundfile 
        # (soundfile is typically needed by librosa for writing)
        import numpy as np
        import soundfile as sf
        
        # 1 sec of pure silence
        dummy_audio = np.zeros(22050, dtype=np.float32) 
        dummy_filename = "dummy_test.wav"
        sf.write(dummy_filename, dummy_audio, 22050)
        
        test_api(dummy_filename)
        
        # Clean up
        import os
        if os.path.exists(dummy_filename):
            os.remove(dummy_filename)
    else:
        # Proceed with actual file passed by user
        file_to_test = sys.argv[1]
        test_api(file_to_test)
