from wasabi import msg

def main(data_dir, folds_dir, mother_out_dir, test, verbose):

    msg.info(f"Initializing training. Using data in: {data_dir}")

    # Name this run.

    with open(data_dir / "info.yaml") as yaml_in:
        data_code = yaml.load(yaml_in)["code"]
    with open(folds_dir / "info.yaml") as yaml_in:
        folds_code = yaml.load(yaml_in)["folds_code"]
    run_name = "crf_run_{}-{}-{}".format(data_code, folds_code, util.four_char_code())
    if test:
        run_name += "_TEST"
    msg.info("Run name: {}", run_name)

    # Create out dir.

    out_dir = mother_out_dir / run_name
    out_dir.mkdir(exist_ok=True)
    msg.info("Writing out to {}", out_dir)

    # Copy data dir and folds dir to out dir, for easy tracking afterwards (but adds heavy data to the output dir as a price.)
    shutil.copytree(data_dir, out_dir / "input" / data_dir.stem)
    shutil.copytree(folds_dir, out_dir / "input" / folds_dir.stem)

    # Do the training.

    folds = _prepare_folded_data(data_dir, folds_dir / "folds.json")
    _train_over_folds(folds, out_dir, test, verbose)

    msg.info("Run {} completed.", run_name)


def _train_over_folds(data_folds: [Fold], out_dir, test, verbose):

    msg.info("Starting cross-validation training.")

    if test:
        msg.warn("Test config: training on only 2 folds.")
        data_folds = data_folds[:2]

    ##~~~~~~~~~~~~~~~~~~~~##
    ## Train all folds.
    ##~~~~~~~~~~~~~~~~~~~~##

    fold_score_dicts = []
    for fold in data_folds:
        # The returned CRF is not used. The score dict is used to average stats over folds.
        _, scores = _train_single_fold(
            fold=fold, fold_out_dir=out_dir / fold.name, test=test, verbose=verbose
        )
        fold_score_dicts.append(scores)

    ## average scores & write results
    # Score dicts take this form:
    # {'label 1': {'precision':0.5,
    #          'recall':1.0,
    #          'f1-score':0.67,
    #          'support':1},
    # 'label 2': { ... },
    # ...
    # }
    #     - as per https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html#sklearn.metrics.classification_report

    def flatten_score_dict(score_dict):
        new = {}
        for label_name, label_scores in score_dict.items():
            for score_name, value in label_scores.items():
                new_key = label_name + "_" + score_name
                new[new_key] = value
        return new

    avg_scores = defaultdict(list)
    for sc in fold_score_dicts:
        sc = flatten_score_dict(sc)
        for k, v in sc.items():
            avg_scores[k].append(v)
    # print(avg_scores)

    with open(out_dir / "avg_scores.json", "w") as o:
        json.dump(avg_scores, o)
    pandas.read_json(out_dir / "avg_scores.json").to_excel(
        out_dir / "collected_scores.xlsx"
    )


def _prepare_folded_data(data_dir, folds_jsonp):

    msg.info("Starting training.")

    ### Read in data. ###

    # The keys of `indexed_X` and `indexed_y` are matching document ID's.
    msg.info("Reading in data.")
    with open(data_dir / "indexed_X.json") as xj:
        indexed_X = json.load(xj)
    with open(data_dir / "indexed_y.json") as yj:
        indexed_y = json.load(yj)

    ### Prepare data folds. ###

    msg.info("Preparing data folds.")

    # Get initial definitions of data folds, that only carry document ids.

    folds = []
    with open(folds_jsonp) as fj:
        # folds_def is a dict with k = fold_id and v = (training_codes, testing_codes)
        folds_def = json.load(fj)
        for fold_id, (train_codes, test_codes) in folds_def.items():
            folds.append(Fold(fold_id, train_codes, test_codes))

    # Add complete docs to each fold.

    for fold in folds:
        for code in fold.train_codes:
            fold.train_X.append(indexed_X[code])
            fold.train_y.append(indexed_y[code])
        for code in fold.test_codes:
            fold.test_X.append(indexed_X[code])
            fold.test_y.append(indexed_y[code])

    return folds


def _train_single_fold(fold: Fold, fold_out_dir, test, verbose):
    """Train a CRF over a single fold, and write out score reports and predictions."""

    msg.info("Fitting fold: {}.", fold.name)

    ##~~~~~~~~~~~~~~~~~~~~##
    ## Initialize CRF.
    ##~~~~~~~~~~~~~~~~~~~~##

    # docs & options = https://sklearn-crfsuite.readthedocs.io/en/latest/api.html#module-sklearn_crfsuite
    if test:
        max_iter = 10
        msg.warn(f"Test config: max iterations capped to {max_iter}.")
        crf = sklearn_crfsuite.CRF(verbose=verbose, max_iterations=max_iter)
    else:
        crf = sklearn_crfsuite.CRF(verbose=verbose)

    ##~~~~~~~~~~~~~~~~~~~~##
    ## Perform fitting and dump the model.
    ##~~~~~~~~~~~~~~~~~~~~##

    crf.fit(X=fold.train_X, y=fold.train_y, X_dev=fold.test_X, y_dev=fold.test_y)
    fold_out_dir.mkdir(exist_ok=True)
    joblib.dump(crf, fold_out_dir / f"model_{fold.name}.pkl")

    ##~~~~~~~~~~~~~~~~~~~~##
    ## Get fold scores and write them out.
    ##~~~~~~~~~~~~~~~~~~~~##

    # CC: I have not found a way to get score directly during model training.
    # So this solution calculates scores by using the model to predict after it's been trained.

    # Get prediction over X.
    predicted_y = crf.predict(fold.test_X)

    # Collect various scores.
    # We use classification reports as provided by `sklearn.metrics`.
    # Lists of lists are not accepted by the sklearn scorer; use this transform the data to the desired format.
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html
    mlb = MultiLabelBinarizer(classes=["I", "O", "B"])
    mlb_gold_y = mlb.fit_transform(fold.test_y)
    mlb_pred_y = mlb.fit_transform(predicted_y)

    # Get the actual reports.
    target_names = ["I", "O", "B"]
    scores_as_text = skl_metrics.classification_report(
        mlb_gold_y, mlb_pred_y, target_names=target_names
    )
    scores_as_dict = skl_metrics.classification_report(
        mlb_gold_y, mlb_pred_y, output_dict=True, target_names=target_names
    )

    # Write scores.
    # with open(fold_out_dir / "scores.yaml", "w") as yaml_out:
    #     yaml_out.write(yaml.dump(scores))  # * For some reason, the formatting is very off when trying to dump as yaml. CC 20/02/2020
    with open(fold_out_dir / "scores.json", "w") as json_out:
        json.dump(scores_as_dict, json_out)
    with open(fold_out_dir / "scores.txt", "w") as txt_out:
        txt_out.write(scores_as_text)

    ##~~~~~~~~~~~~~~~~~~~~##
    ## Get fold predictions and write them out.
    ##~~~~~~~~~~~~~~~~~~~~##

    _write_predictions(
        crf,
        gold_X=fold.test_X,
        gold_y=fold.test_y,
        predictions_outp=fold_out_dir / "predictions.txt",
        info_outp=fold_out_dir / "info.yaml",
    )

    return crf, scores_as_dict


def _write_predictions(crf, gold_X, gold_y, predictions_outp, info_outp):
    """Make predictions over the given `y` data and write them to `outp`, comparing them to gold output."""

    # aux
    def pretty_string(sentence_as_tags, sentence_as_tokens):
        s = []

        def tag_s(token, tag):
            if tag == "O":
                return "".join(("." if i == 0 else " ") for i, _ in enumerate(token))
            else:
                return "".join((tag if i == 0 else "-") for i, _ in enumerate(token))

        for tag, token in zip(sentence_as_tags, sentence_as_tokens):
            s.append(tag_s(token, tag))

        return " ".join(s)

    # make predictions
    correct_predictions = []
    incorrect_predictions = []
    for gold_X_sentence, gold_y_sentence in zip(gold_X, gold_y):
        # TODO predict all, rather than single?
        predicted_y_sentence = crf.predict_single(gold_X_sentence)

        if predicted_y_sentence == gold_y_sentence:
            correct_predictions.append((gold_y_sentence, predicted_y_sentence))
        else:
            incorrect_predictions.append(
                (gold_X_sentence, gold_y_sentence, predicted_y_sentence)
            )

    # write out incorrect predictions
    with open(predictions_outp, "w") as o:
        for gold_X, gold_y, pred_y in incorrect_predictions:
            tokens = [tok["token"] for tok in gold_X]
            line = "\n\nTokens:\t\t{}\nGold:\t\t{}\nPred:\t\t{}".format(
                " ".join(tokens),
                pretty_string(gold_y, tokens),
                pretty_string(pred_y, tokens),
            )
            o.write(line)

    info = {
        "n_correct_predictions": len(correct_predictions),
        "n_incorrect_predictions": len(incorrect_predictions),
    }
    with open(info_outp, "w") as o:
        o.write(yaml.dump(info))
