import pickle
import json
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from towhee import pipeline



def read_concept(data):
    concept_dict = {}
    # 'Fashion': 49282, 'Travel&Active&Sports': 66469,
    # 'Entertainment': 29590, 'Holiday&Celebrations': 54751, 'Food': 16727,
    # 'Whether&Season': 20292, 'Animal': 19992, 'Family': 4274, 'Social&People': 23351,
    # 'Urban': 15272, 'Electronics': 5613}
    with open(data, 'r') as f:
        text = json.load(f)
        for img_dict in text:
            pid = img_dict['Pid']
            concept = img_dict['Concept']
            concept_dict[pid] = concept
    return concept_dict


def text_embedding(data):
    with open('../data/SMP/pid2idx.pickle', 'rb') as f:
        pid2idx = pickle.load(f)
    text_embedding_list = [[]] * 305613
    model = SentenceTransformer('all-MiniLM-L6-v2').cuda()
    with open(data, 'r') as f:
        text = json.load(f)
        for img_dict in tqdm(text):
            pid = img_dict['Pid']
            title = img_dict['Title']
            title_embedding = model.encode(title)
            text_embedding_list[pid2idx[int(pid)]] = title_embedding
    with open('../data/SMP/text_embedding_list.pickle', 'wb') as fwrite:
        pickle.dump(text_embedding_list, fwrite)
    return text_embedding_list

def img_embedding():
    with open('../data/SMP/pid2idx.pickle', 'rb') as f:
        pid2idx = pickle.load(f)
    cate_dict = read_concept('../data/SMP/train_category.json')
    img_embedding_dict = {}
    with open('../data/SMP/train_category.json', 'r') as f:
        text = json.load(f)
        for line in tqdm(text):
            pid = line['Pid']
            uid = line['Uid']
            category = line['Category']
            concept = line['Concept']
            img_path = '../data/SMP/train_img/' + uid + '/' + pid + '.jpg'
            embedding_pipeline = pipeline('towhee/image-embedding-resnet50')
            embedding = embedding_pipeline(img_path)
            img_embedding_dict[pid2idx[int(pid)]] = [embedding, concept]
    print(img_embedding_dict)
    with open('../data/SMP/img_embedding.pickle', 'wb') as fwrite:
        pickle.dump(img_embedding_dict, fwrite)
    return img_embedding_dict

text_embedding('../data/train_text.json')


