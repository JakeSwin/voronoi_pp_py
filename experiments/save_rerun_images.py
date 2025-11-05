import rerun as rr
import pandas as pd
from PIL import Image

folder = "/home/swin/Documents/rerun/"

recording = rr.dataframe.load_recording(f"{folder}online_gp_0.rrd")

view = recording.view(index="log_time", contents="GP/TestMap")
table = view.select().read_all()
df = table.to_pandas()

print(df)
print(df.columns.tolist())
format = df["/GP/TestMap:Image:format"][0][0]
print(format)
for idx, msg in enumerate(df["/GP/TestMap:Image:buffer"]):
    arr = msg[0].reshape(format["width"], format["height"], 4)
    img = Image.fromarray(arr).convert("L")
    img.save(f"{folder}{idx}_image.png")
    # print(img.shape)

# print(format)
