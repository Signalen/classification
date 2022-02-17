import argparse
import json
from engine import TextClassifier
from django.utils.text import slugify

def parse_args():
    parser = argparse.ArgumentParser()
    optional = parser._action_groups.pop() 
    required = parser.add_argument_group('required arguments')
    required.add_argument('--csv', required=True)
    optional.add_argument('--columns', default='')
    optional.add_argument('--fract', default=1.0, type=float)
    optional.add_argument('--output-fixtures', const=True, nargs="?", default=True, type=bool)
    optional.add_argument('--output-validation', const=True, nargs="?", default=False, type=bool)
    parser._action_groups.append(optional)
    return parser.parse_args()

def generate_category(slug, name, pk, fk, parent_name=None):
    return {
        "model": "signals.category",
        "pk": pk,
        "fields": {
            "parent": fk,
            #"slug": slugify(name if not parent_name else "{main}-{sub}".format(main=parent_name, sub=name)),
            "slug": slugify(slug),
            "name": name,
            "handling": 'REST',
            "handling_message": None,
            "is_active": True,
            "description": None
        }
    }

def generate_fixtures(categories):
    cats = {}
    idx = 9991

    # add global overig
    categories.append(['overig', 'overig'])

    # process parents first
    for cat in categories:
        slug = slugify(cat[0])
        if not slug in cats:
            parent = generate_category(slug=cat[0], name=cat[0], pk=idx, fk=None)
            cats[slug] = parent
            idx = idx + 1
            # generate overig
            if slug != 'overig':
                cats["{}|Overig".format(slug)] = generate_category(
                    slug="overig",
                    name="Overig {}".format(cat[0]),
                    pk=idx,
                    fk=parent["pk"])
                idx = idx + 1

    # process childs
    for cat in categories:
        parent_slug = slugify(cat[0])
        slug = "{main}|{sub}".format(main=parent_slug, sub=slugify(cat[1]))
        if not slug in cats:
            parent = cats[parent_slug]
            cats[slug] = generate_category(slug=cat[1], name=cat[1], pk=idx, fk=parent["pk"], parent_name=cat[0])
            idx = idx + 1

    # validate slug length
    for cat in cats:
        slug = cats[cat]['fields']['slug']
        if len(slug) > 50:
            print("Warning invalid slug: {slug}, length: {length}".format(slug=slug, length=len(slug)))

    return cats.values()
        
def train(df, columns, output_validation=False, output_fixtures=True):
    texts, labels, train_texts, train_labels, test_texts, test_labels = classifier.make_data_sets(df, columns=columns)
    colnames = "_".join(columns)
    df.to_csv("{}_dl.csv".format(colnames), mode='w', columns=['Text','Label'], index=False)
    print("Training... for columns: {colnames}".format(colnames=colnames))
    model = classifier.fit(train_texts, train_labels)
    print("Serializing model to disk...")
    classifier.export_model("{file}_model.pkl".format(file=colnames))
    if len(columns) > 1:
        cats = [x.split('|') for x in model.classes_] 
        # slugs = ["/categories/{main}/sub_categories/{main}-{sub}".format(main=slugify(x[0]), sub=slugify(x[1])) for x in cats]
        slugs = ["/categories/{main}/sub_categories/{sub}".format(main=slugify(x[0]), sub=slugify(x[1])) for x in cats]
        if output_fixtures:
            fixtures = generate_fixtures(cats)
            with open("{file}_fixtures.json".format(file=colnames), 'w') as outfile:
                json.dump(list(fixtures), outfile)
    else:
        slugs = ["/categories/{main}".format(main=slugify(x)) for x in model.classes_]
    classifier.pickle(slugs, "{file}_labels.pkl".format(file=colnames))
    
    print("Validating model")
    test_predict, precision, recall, accuracy = classifier.validate_model(
        test_texts,
        test_labels,
        "{colnames}-matrix.pdf".format(colnames=colnames),
        "{colnames}-matrix.csv".format(colnames=colnames),
        dst_validation = "{colnames}_validation.csv".format(colnames=colnames) if output_validation else None)
    print('Precision', precision)
    print('Recall', recall)
    print('Accuracy', accuracy)


if __name__ == '__main__':
    args = parse_args()
    print("Using args: {}".format(args))

    classifier = TextClassifier()
    df = classifier.load_data(csv_file=args.csv, frac=args.fract)
    if len(df) == 0:
        print("Failed to load {}".format(args.csv))
        exit(-1)
    else:
        print("{} rows loaded".format(len(df)))
    texts, labels, train_texts, train_labels, test_texts, test_labels = classifier.make_data_sets(df)
    columns = args.columns or 'Main'
    print("Training using category column(s): {}".format(columns))
    # train sub cat
    train(df, columns.split(','), args.output_validation, args.output_fixtures)
