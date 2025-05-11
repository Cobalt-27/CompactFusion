test_method="binary"
python extract_video.py --video "video/${test_method}.mp4" --output keyframes
python image_collage.py --output "stack/${test_method}.jpg"
mv keyframes "keyframes_${test_method}"