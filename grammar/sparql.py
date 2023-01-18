
import json
import pickle
import requests
import networkx as nx
from copy import deepcopy
from typing import List
from networkx import DiGraph

def format_sparql(sparql: str) -> str:
    url = "http://localhost:8080/format"
    data = {"sparql": sparql}
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, data=json.dumps(data), headers=headers)
    response = json.JSONDecoder().decode(response.text)["data"]
    return response

class SparqlGrammar(object):

    def __init__(self) -> None:

        self.sparql_grammar_graph = pickle.load(open("./grammar.pickle", "rb"))
        self.select_grammar_graph = pickle.load((open("./select_grammar.pickle", "rb")))
        self.ask_grammar_graph = pickle.load((open("./ask_grammar.pickle", "rb")))
        self.where_grammar_graph = pickle.load((open("./where_grammar.pickle", "rb")))
        self.solution_grammar_graph = pickle.load((open("./solution_grammar.pickle", "rb")))

        self.intent = ""

        self.initialize()

    def initialize(self) -> None:

        nodes = deepcopy(self.sparql_grammar_graph.nodes)
        nodes = sorted(nodes)

        self.keyword2id = {k: v for k, v in zip(nodes, range(len(nodes)))}
        self.id2keyword = {v: k for k, v in self.keyword2id.items()}


    def keywords(self) -> List:
        return list(self.keyword2id.keys())

    def get_possible_next_keywords(self, sparql):

        if sparql == "":
            return ["SELECT", "ASK"]

        last_word = sparql.split()[-1]

        if "SELECT" in sparql:
            self.intent = "SELECT"
        elif "ASK" in sparql:
            self.intent = "ASK"

        return self.extract_successors(last_word)


    def extract_successors(self, node):
        if self.intent == "SELECT":
            return [edge[1] for edge in self.select_grammar_graph.out_edges(node)]
        elif self.intent == "ASK":
            return [edge[1] for edge in self.ask_grammar_graph.out_edges(node)]

    def _validate(self, sequence: str) -> bool:
        stack = []
        opening = set('([{')
        closing = set(')]}')
        pair = {')': '(', ']': '[', '}': '{'}
        for i in sequence:
            if i in opening:
                stack.append(i)
            if i in closing:
                if not stack:
                    return False
                elif stack.pop() != pair[i]:
                    return False
                else:
                    continue
        if not stack:
            return True
        else:
            return False

    def _format(self, sparql) -> str:
        return format_sparql(sparql)

    def parseQuery(self, query: str):
        pass



if __name__ == '__main__':

    sparql_grammar = SparqlGrammar()

    sparql = ""
    print(sparql_grammar.get_possible_next_keywords(sparql))
    while True:
        inp = input(f"Please enter: {sparql} ")
        if sparql == "":
            sparql = inp
        else:
            sparql += " "+inp
        print(sparql_grammar.get_possible_next_keywords(sparql))

