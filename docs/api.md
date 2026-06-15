# API reference

Re-exported at the top level:
`from atombench import submit, validate_submission, SubmissionError`.

## Submitting to the leaderboard

```{eval-rst}
.. autofunction:: atombench.submit.submit
```

```{eval-rst}
.. autofunction:: atombench._leaderboard.validate_submission
```

```{eval-rst}
.. autoclass:: atombench._leaderboard.ValidationReport
   :members:
```

```{eval-rst}
.. autoexception:: atombench._leaderboard.SubmissionError
```

## Computing metrics

```{eval-rst}
.. autofunction:: atombench.cli.compute_metrics
```

## Verifying benchmark consistency

`atombench-verify <path>` checks that benchmark CSVs for the same dataset share an
identical test-set ID list (and warns if `target` structures disagree). Building
blocks:

```{eval-rst}
.. autofunction:: atombench.verify.group_by_dataset
```

```{eval-rst}
.. autofunction:: atombench.verify.compare_group
```
