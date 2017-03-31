from iepy.data.models import Relation, EntityKind


if __name__ == "__main__":
    company = EntityKind.objects.get_or_create(name="company")[0]
    relation = EntityKind.objects.get_or_create(name="relation")[0]
    Relation.objects.get_or_create(name="relation", left_entity_kind=company, right_entity_kind=relation)