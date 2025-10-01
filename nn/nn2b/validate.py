def validate_predictions(predict_output, scrap_output):
    predict_set = set(v[1] for v in predict_output)
    assert set(list(range(26))) == predict_set

    if scrap_output:
        scrap_set = set(v[1] for v in scrap_output)
        assert set(list(range(26))) == scrap_set
