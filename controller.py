# coding:utf-8
from flask import Flask, render_template, request, redirect, url_for, make_response, jsonify
import json
import os
import time
import base64
from scipy.misc import imsave
import torch
from datetime import timedelta

# SketchTriplet network--------------------
from SketchTriplet.SketchTriplet_half_sharing import BranchNet
from SketchTriplet.SketchTriplet_half_sharing import SketchTriplet as SketchTriplet_hs
from SketchTriplet.flickr15k_dataset import flickr15k_dataset_lite
from SketchTriplet.retrieval import retrieval


def load_model_retrieval():
    net_dict_path = './static/model/500.pth'
    branch_net = BranchNet()  # for photography edge
    net = SketchTriplet_hs(branch_net)
    net.load_state_dict(torch.load(net_dict_path))
    net = net.cuda()
    net.eval()
    return net


# Retrieval model and dataset definition
flickr15k_dataset = flickr15k_dataset_lite()
retrieval_net = load_model_retrieval()

# SketchAPP definition
app = Flask(__name__, template_folder='templates', static_folder='static')
app.send_file_max_age_default = timedelta(seconds=1)


@app.route('/canva', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        sketch_src = request.form.get("sketchUpload")
        upload_flag = request.form.get("uploadFlag")
        sketch_src_2 = None
        if upload_flag:
            sketch_src_2 = request.files["uploadSketch"]
        if sketch_src:
            flag = 1
        elif sketch_src_2:
            flag = 2
        else:
            return render_template('canva.html')

        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(basepath, 'static/sketch_tmp', 'upload.png')
        if flag == 1:
            # base64 image decode
            sketch = base64.b64decode(sketch_src[22:])
            user_input = request.form.get("name")
            file = open(upload_path, "wb")
            file.write(sketch)
            file.close()

        elif flag == 2:
            # upload sketch
            sketch_src_2.save(upload_path)
            user_input = request.form.get("name")

        # for retrieval
        retrieval_list, real_path = retrieval(retrieval_net, upload_path, flickr15k_dataset)
        real_path = json.dumps(real_path)

        return render_template('panel.html', userinput=user_input, val1=time.time(), upload_src=sketch_src,
                               retrieval_list=retrieval_list,
                               json_info=real_path)

    return render_template('canva.html')


@app.route('/')
def homepage():
    return render_template('canva.html')


if __name__ == '__main__':
    # open debug mode
    app.run(debug=True)
