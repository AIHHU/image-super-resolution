import paddle
import paddle.fluid as fluid
import numpy as np
from PIL import Image


def load_image(dirname):
    a=Image.open(dirname)
    a=np.array(a).astype("float32").transpose([2,0,1])/255
    a=a.reshape(1,a.shape[0],a.shape[1],a.shape[2])
    return a

def inference_image(dirname,infer_program,feeded_var_names,target_var):
    image=load_image(dirname)
    output=exe.run(program=infer_program,feed={feeded_var_names[0]:image},fetch_list=target_var)
    output_image=output[0]
    return output_image

save_path="best_model"
image_road="finger.jpg"
save_name="finger.npy"
paddle.enable_static()


place=fluid.CPUPlace()
exe=fluid.Executor(place)
exe.run(fluid.default_startup_program())
[infer_program,feeded_var_names,target_var]=fluid.io.load_inference_model(dirname=save_path,executor=exe)

result=inference_image(image_road, infer_program, feeded_var_names, target_var)
np.save(save_name,result)