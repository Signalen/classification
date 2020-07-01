import argparse
from engine import TextClassifier
from django.utils.text import slugify

def parse_args():
    parser = argparse.ArgumentParser()
    optional = parser._action_groups.pop() 
    required = parser.add_argument_group('required arguments')
    required.add_argument('--csv', required=True)
    optional.add_argument('--columns', default='')
    optional.add_argument('--ml-prefix', default='https://api.data.amsterdam.nl/signals/v1/public/terms/categories')
    optional.add_argument('--fract', default=1.0, type=float)
    optional.add_argument('--output-validation', const=True, nargs="?", default=False, type=bool)
    parser._action_groups.append(optional)
    return parser.parse_args()

def train(df, columns, prefix, output_validation=False):
    texts, labels, train_texts, train_labels, test_texts, test_labels = classifier.make_data_sets(df, columns=columns)
    colnames = "_".join(columns)
    df.to_csv("{}_dl.csv".format(colnames), mode='w', columns=['Text','Label'], index=False)
    print("Training... for columns: {colnames}".format(colnames=colnames))
    model = classifier.fit(train_texts, train_labels)
    print("Serializing model to disk...")
    classifier.export_model("{file}_model.pkl".format(file=colnames))
    if len(columns) > 1:
        cats = [x.split('|') for x in model.classes_] 
        slugs = ["{prefix}/categories/{main}/sub_categories/{sub}".format(prefix=prefix, main=slugify(x[0]), sub=slugify(x[1])) for x in cats]
    else:
        slugs = ["{prefix}/categories/{main}".format(prefix=prefix, main=slugify(x)) for x in model.classes_]
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
    train(df, columns.split(','), args.ml_prefix, args.output_validation)
