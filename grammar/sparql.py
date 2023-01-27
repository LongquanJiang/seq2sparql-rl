import lark
from lark import Lark, Transformer, Tree, Token
from lark.grammar import NonTerminal, Terminal

sparql_grammar = open("simplified_sparql.lark", "r").read()
sparql_lark = Lark(sparql_grammar, start="start", parser="earley", keep_all_tokens=True)

generated_query = []

multipleoptions_rules = ["query", "varorexpressionasvarrepeatorasterisk", "groupcondition", "constraintorvar", "ordercondition", "ascordesc", "constraint",
                         "graphpatternnottriples", "expressionlist", "varorterm", "varoriri", "graphterm", "enlmlminnumericalexpression", "pmnmultiplicativeexpression",
                         "asteriskorslashunaryexpression", "unaryexpression", "primaryexpression", "builtincall", "aggregate", "asteriskorexpression", "booleanliteral"]

optional_rules = ["distinct_optional",
                  "where_optional",
                  "groupclause_optional",
                  "orderclause_optional",
                  "limitclause_optional",
                  "asvar_optional",
                  "triplesblock_optional",
                  "dot_optional",
                  "dottriplesblockoptional_optional",
                  "enlmlminnumericalexpression_optional",
                  "pmnmultiplicativeexpression_optional"]

plus_rules = ["expressionasvar",
              "groupcondition",
              "ordercondition"]

star_rules = ["graphpatternnottriplesdottriplesblock_star",
              "commaexpression_star",
              "logicorconditionalandexpression_star",
              "logicandvaluelogical_star",
              "pmnmultiplicativeexpressionoptional_star",
              "asteriskorslashunaryexpression_star"]


def find_first_non_terminal(query):

    for idx, q in enumerate(query):

        if isinstance(q, NonTerminal):
            return idx, q

    return -1, None


def flatten_grammar_rule(rule_def):

    rtn = []
    children = [rule_def[2]]
    while len(children) != 0:

        pointer = children.pop(0)

        if pointer.data == "value":
            name_pointer = pointer.children[0]
            if isinstance(name_pointer, Tree):
                if name_pointer.data == "literal":
                    rtn.append(name_pointer.children[0])
            elif isinstance(name_pointer, NonTerminal):
                rtn.append(name_pointer)
        else:
            children = pointer.children + children
    return rtn


def get_grammar_rule_by_name(name):
    rule_defs = sparql_lark.grammar.rule_defs
    for rule_def in rule_defs:
        if rule_def[0] == name:
            return rule_def

def get_node_from_rule_by_name(rule, name):
    for node in rule:
        if node.name == name:
            return node
    return None


def print_sparql_query(query):
    for q in query: print(q)

if __name__ == '__main__':

    time_step = -1

    while True:

        time_step += 1

        if time_step == 0:
            initial_rule_def = get_grammar_rule_by_name("start")
            initial_rule = flatten_grammar_rule(initial_rule_def)
            generated_query = initial_rule
            print(f"\nTime Step {time_step}: ")
            print_sparql_query(generated_query)
            continue

        idx, action = find_first_non_terminal(generated_query)

        if idx == -1: # terminate the loop if there is no any non-terminal
            break

        if action.name in optional_rules or action.name in plus_rules or action.name in star_rules or action.name in multipleoptions_rules:

            current_rule_def = get_grammar_rule_by_name(action.name)
            current_rule = flatten_grammar_rule(current_rule_def)

            print(f"\nCurrent rule: {current_rule}")
            if action.name in optional_rules or action.name in multipleoptions_rules:
                if action.name in multipleoptions_rules:
                    inp = input("Please choose a grammar rule: ").strip()
                    node = get_node_from_rule_by_name(current_rule, inp)
                    generated_query = generated_query[0:idx] + [node] + generated_query[idx + 1:]
                else:
                    inp = input("Do you keep it or not? (yes/no): ").strip()
                    if inp == "no":
                        generated_query = generated_query[0:idx] + generated_query[idx+1:]
                    else:
                        generated_query = generated_query[0:idx] + current_rule + generated_query[idx + 1:]

            elif action.name in plus_rules:
                inp = input("Repeat or not repeat this grammar rule? (yes/no): ").strip()
                if inp == "yes":
                    generated_query = generated_query[0:idx+1] + [action] + generated_query[idx+1:]
                    generated_query = generated_query[0:idx] + current_rule + generated_query[idx + 1:]
                elif inp == "no":
                    generated_query = generated_query[0:idx] + current_rule + generated_query[idx+1:]
            elif action.name in star_rules:
                inp = input("Repeat, not repeat or make it empty? (repeat/notrepeat/nil): ").strip()
                if inp == "repeat":
                    generated_query = generated_query[0:idx+1] + [action] + generated_query[idx+1:]
                    generated_query = generated_query[0:idx] + current_rule + generated_query[idx + 1:]
                elif inp == "notrepeat":
                    generated_query = generated_query[0:idx] + current_rule + generated_query[idx+1:]
                elif inp == "nil":
                    generated_query = generated_query[0:idx] + generated_query[idx+1:]
        else:
            new_rule_def = get_grammar_rule_by_name(action.name)
            new_rule = flatten_grammar_rule(new_rule_def)
            generated_query = generated_query[0:idx] + new_rule + generated_query[idx+1:]

        print(f"\nTime Step {time_step}:")
        print_sparql_query(generated_query)
    print(f"\nTime Step {time_step}:")
    print_sparql_query(generated_query)