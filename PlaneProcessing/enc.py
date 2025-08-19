import rosbag
from sensor_msgs.msg import Image

bag = rosbag.Bag('data_1.bag', 'r')

encoding = None

for topic, msg, t in bag.read_messages(topics='/camera/color/image_raw'):
    if isinstance(msg, Image):
        encoding = msg.encoding
        break

bag.close()

if encoding is not None:
    print(f"The encoding for /camera/color/image_raw is: {encoding}")
else:
    print("No messages found on the /camera/color/image_raw topic.")

