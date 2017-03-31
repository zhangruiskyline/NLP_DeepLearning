from refo import Question, Star, Any, Plus
from iepy.extraction.rules import rule, Token, Pos

RELATION = "Acquisition"

@rule(True)
def company_relationship(Subject, Object):
    """
    Ex: Gary Sykes (Born 13 February 1984) is a British super featherweight boxer.
    """
    anything = Star(Any())
    born = Star(Pos(":")) + Question(Token("Bought") | Token("buy")) + Question(Token("c."))
    entity_leftover = Star(Pos("NNP"))
    return Subject + entity_leftover + Pos("-LRB-") + born + Object + Pos("-RRB-") + anything