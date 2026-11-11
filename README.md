# Documentation evidence auditor

An offline audit tool that compares exact documentation wording with source patterns. It reads the root README and Markdown files below any docs directory, then inspects source syntax trees. It uses only the standard library, never executes inspected code, and has no network access or telemetry.

Install and run:

```sh
pip3 install .
text-classification-claim-audit --version
text-classification-claim-audit audit . --json
python3 -m text_classification_claim_audit audit . --verbose
text-classification-claim-audit audit . --fail-on medium --ignore vendor --ignore build
python3 -m unittest discover
```

The default threshold is high. Exit status 0 means no finding reaches the threshold; 2 means at least one does; 1 indicates an input or usage error. A medium finding can therefore appear with status 0. JSON includes verdict, findings with rule, severity, message and paths, plus production and test evidence paths. The verdict reports whether findings exist independently of the exit threshold.

The rules are:

```text
documented-text-classification-claim-false HIGH
documented-text-classification-claim-test-only MEDIUM
Supporting evidence exists only in tests.
documented-no-text-classification-claim-false HIGH
```

Fenced examples are omitted from prose checks; inline code counts. Symlinks are skipped. Source below docs, fixtures and examples is excluded; tests are considered separately. Ignore options each accept one relative path prefix.

## Differentiation

This auditor checks one specific family of promises and source patterns. Other checkers cover the following adjacent operations; their patterns alone do not satisfy this auditor:

```text
whole-image labelling
sentiment analysis
topic modelling
summarization
translation
tokenization
text generation
```

## Honest limitations

Static pattern matching is not semantic judgement. Dead code counts. Alias chains the checker cannot follow may be missed. Wording checks use exact phrases, not semantics; negation depends on the negation phrase list. Pairing must be in the same module. A recognized pattern does not prove runtime correctness, model quality, or that documentation accurately describes every behavior.

Version 0.1.0. Distributed under the MIT license.
