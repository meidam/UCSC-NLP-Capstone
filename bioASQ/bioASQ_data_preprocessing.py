import json
import bs4 as bs
import urllib.request
from tqdm import tqdm

# will output dataset in the exact same format as Covid-QA and Squad
def bioASQ_preprocessing():
  filename = 'training9b.json'
  training = open(filename, 'r')
  # bioASQ_dict['data'][:]{'paragraphs': [{'qas': {'question':..., 'id':..., 'answers': [{'text':..., 'answer_start': ...}], 'is_imposible': ...}}], 'context':...}
  bioASQ = training.read()
  training.close()

  bioASQ_dict = {}
  data_arr = []
  data = json.loads(bioASQ)
  errs = 0
  num_exs = 0
  for questions in tqdm(data['questions']):
    if questions['type'] == 'factoid':
      snippets = questions['snippets']
      ideal = questions['ideal_answer']
      for sub in snippets:
        num_exs += 1
        try:
          qas = dict()
          source = urllib.request.urlopen(sub['document'])
          soup = bs.BeautifulSoup(source, 'lxml')
          context = soup.p.get_text()

          qas['answers'] =  [{'answer_start':sub['offsetInBeginSection'],
                              'text': sub['text']}]
          qas['id'] = questions['id']
          qas['is_imposible'] = False
          qas['question'] = questions['body']
          
          #print(sub['document']) # this is the link to the website that has the link
          #print(sub['beginSection']) idk what this is

          data_arr.append({'paragraphs': [{'qas':[qas], 'context': context}]})
        except:
          errs += 1
          try:
            source = urllib.request.urlopen(sub['document'])
            print('url works fine, there is an issue in the data processing')
          except:
            print(sub['document'], ' was not found')

  bioASQ_dict['data'] = data_arr
  return bioASQ_dict