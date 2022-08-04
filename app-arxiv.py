from flask_api import status
from flask_cors import CORS
from flask import Flask,request
import pathlib
import torch
from preprocess import preprocess_pegasus, post_process_output_bigbird, remove_last_scentence_arvix


app = Flask(__name__)
CORS(app)
current_path = str(pathlib.Path(__file__).parent.absolute())
from transformers import BigBirdPegasusForConditionalGeneration, AutoTokenizer

import re
from transformers import BartTokenizer, BartForConditionalGeneration
torch_device = torch.device("cuda:0")
import helpers
modelArxiv = BigBirdPegasusForConditionalGeneration.from_pretrained("google/bigbird-pegasus-large-arxiv").to("cuda")
tokenizerArxiv = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-arxiv")

global is_status
is_status = True
@app.route('/')
def GetStatusService():
    return 'ok'


@app.route('/BigBirdArxiv', methods=['POST'])
def post1():
    if request.method =="POST":
        content = request.get_json()
        result = {}
        result['Model'] = 'BigBirdArxiv'
        result['Summary'] = ''
        if content['text'] is not None:
            docs = helpers.split_doc(content['text'], 4096)
            sum_text= ''
            for doc in docs:
                doc = preprocess_pegasus(doc)
                tokens = tokenizerArxiv(doc, return_tensors='pt', truncation=True, max_length=4096).to("cuda")
                summary = modelArxiv.generate(**tokens)
                tgt_text = tokenizerArxiv.batch_decode(summary, no_repeat_ngram_size=3) # skip_special_tokens=True, clean_up_tokenization_spaces=False,
                sum_text += remove_last_scentence_arvix(post_process_output_bigbird(tgt_text[0])) + '\n'
            result['Summary'] = sum_text
            torch.cuda.empty_cache()
            print(result)
            return  result
        return result
    return None
@app.route('/change_status', methods=['POST'])
def post2():
    if request.method =="POST":
        content = request.get_json()
        response_change_status ={}
        response_change_status['result'] = False
        global is_status
        try:
            if content['status'] == True and is_status == False:
                # khởi tạo mô hình
                is_status = True
            elif content['status'] == False and is_status == True:
                # xóa mô hình
                is_status = False
            else:
                print('ok')
            response_change_status['result'] = True
        except:
            response_change_status['result'] = False
    return response_change_status

@app.route('/get_status')
def get():
    global is_status
    response_status ={}
    response_status['status'] = is_status
    return response_status
app.run(host='0.0.0.0', port=5200,threaded=True)
