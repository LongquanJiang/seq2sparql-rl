import sys
sys.path.append("..")

import json
import argparse

"""
Get recognized entities and relations per question 
"""

import json
import requests

def call_falcon_2(sentence):
    url = "https://labs.tib.eu/falcon/falcon2/api"
    data = {"text": sentence}
    params = {"mode": "long"}
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, params=params, data=json.dumps(data), headers=headers)
    response = json.JSONDecoder().decode(response.text)
    return response


def get_Falcon2_predictions(question_string):
    predictions = call_falcon_2(question_string)

    entities = {"entities_wikidata": [entity["URI"].rsplit("/", 1)[1] for entity in predictions["entities_wikidata"]]} if predictions.get("entities_wikidata") else {}
    relations = {"relations_wikidata": [relation["URI"].rsplit("/", 1)[1] for relation in predictions["relations_wikidata"]]} if predictions.get("relations_wikidata") else {}

    entity_surface_forms = {entity["URI"].rsplit("/", 1)[1]: entity["surface form"] for entity in predictions["entities_wikidata"]} if predictions.get("entities_wikidata") else {}
    relation_surface_forms = {relation["URI"].rsplit("/", 1)[1]: relation["surface form"] for relation in predictions["relations_wikidata"]} if predictions.get("relations_wikidata") else {}

    entities_relations = {**entities, **relations}
    surface_forms = {**entity_surface_forms, **relation_surface_forms}
    return entities_relations, surface_forms


def retrieve_ent_rel(data):

    entities_relations = dict()
    surface_forms = dict()

    idx = 0
    for question_id, question_string in data.items():

        q_entities_relations, q_surface_forms = get_Falcon2_predictions(question_string)

        if question_id not in entities_relations:
            entities_relations[question_id] = q_entities_relations

        surface_forms.update(q_surface_forms)

        idx += 1
        print("Processed: {0}, {1}/{2}: ".format(question_id, idx, len(data)))

    return entities_relations, surface_forms


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str, help="the dataset name to be processed.")

    args = parser.parse_args()

    if args.dataset == "lcq2":  # wikidata

        train_data = json.load(open("../data/LC-QuAD2/train_data/all_questions_trainset.json"))
        train_entities_relations, train_surface_forms = retrieve_ent_rel(train_data)
        with open("../data/LC-QuAD2/train_data/all_entities_relations_trainset.json", "w") as output_file:
            json.dump(train_entities_relations, output_file)

        test_data = json.load(open("../data/LC-QuAD2/test_data/all_questions_testset.json"))
        test_entities_relations, test_surface_forms = retrieve_ent_rel(test_data)
        with open("../data/LC-QuAD2/test_data/all_entities_relations_testset.json", "w") as output_file:
            json.dump(test_entities_relations, output_file)