import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

app = FaceAnalysis(allowed_modules=['detection'], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                   download=False)
app.prepare(ctx_id=0, det_size=(640, 640))
img = ins_get_image('t1')  # 不用带后缀，图片放到./insightface/python-package/insightface/data/images
faces = app.get(img)
print("faces::::", faces)
print("len:", len(faces))
rimg = app.draw_on(img, faces)
cv2.imwrite("./ldh_output.jpg", rimg)
cv2.imshow("frame", rimg)
if cv2.waitKey(0) & 0xFF == ord('Q'):
    cv2.destroyAllWindows()