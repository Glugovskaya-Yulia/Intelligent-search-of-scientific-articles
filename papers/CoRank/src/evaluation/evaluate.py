from ranx import evaluate

def evaluate_runs(qrels, runs, metrics):
    results = {}
    for name, run in runs.items():
        results[name] = evaluate(
            qrels=qrels,
            run=run,
            metrics=metrics,
        )
    return results
