from flask_api import status
from flask_cors import CORS
from flask import Flask,request
import pathlib
import torch
from preprocess import preprocess_pegasus, post_process_output_bigbird


app = Flask(__name__)
CORS(app)
current_path = str(pathlib.Path(__file__).parent.absolute())
from transformers import BigBirdPegasusForConditionalGeneration, AutoTokenizer

import re
from transformers import BartTokenizer, BartForConditionalGeneration
torch_device = torch.device("cuda:0")
import helpers


modelPubmed = BigBirdPegasusForConditionalGeneration.from_pretrained("google/bigbird-pegasus-large-pubmed",attention_type="original_full").to(torch_device)
tokenizerPubmed = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-pubmed")
global is_status
is_status = True

@app.route('/')
def GetStatusService():
    return 'ok'



@app.route('/BigBirdPubmed', methods=['POST'])
def post3():
    if request.method =="POST":
        content = request.get_json()
        result = {}
        result['Model'] = 'BigBirdPubmed'
        result['Summary'] = ''
        if content['text'] is not None:
            content['text'] = preprocess_pegasus(content['text'])
            docs = helpers.split_doc(content['text'], 4096)
            sum_text= ''
            for doc in docs:
                tokens = tokenizerPubmed(doc, truncation=True, padding="longest", return_tensors="pt").to(torch_device)
                summary = modelPubmed.generate(**tokens)
                tgt_text = tokenizerPubmed.decode(summary[0], skip_special_tokens=True)
                sum_text += post_process_output_bigbird(tgt_text) + '\n'
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

app.run(host='0.0.0.0', port=5300,threaded=True)
