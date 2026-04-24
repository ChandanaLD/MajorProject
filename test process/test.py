from my_models import extract_faces_from_video
import cv2

frames = extract_faces_from_video("C:\\Users\\chand\\Downloads\\videoplayback (2).mp4")

print("Frames extracted:", len(frames))
for i, f in enumerate(frames):
    cv2.imwrite(f"debug_frame_{i}.jpg", f)
