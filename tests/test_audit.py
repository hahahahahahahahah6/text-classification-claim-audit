import unittest
import ast
from text_classification_claim_audit import phrases

class PhraseOracle(unittest.TestCase):
    def test_exact_vocabulary(self):
        from pathlib import Path
        parsed = ast.parse(Path(phrases.__file__).read_text())
        literals = {n.targets[0].id: ast.literal_eval(n.value) for n in parsed.body if isinstance(n, ast.Assign)}
        self.assertEqual(literals["TEXTCLASSIFICATION"], ('text classification', 'text classifier', 'text classifiers', 'classifies text', 'classify text', 'classify texts', 'classifies texts', 'classifying text', 'text categorization', 'text categorisation', 'categorizes text', 'categorizing text', 'document classification', 'document classifier', 'document classifiers', 'classifies documents', 'classify documents', 'classifying documents', 'document categorization', 'categorizes documents', 'spam detection', 'detects spam', 'spam filtering', 'spam filter', 'spam classifier', 'classifies spam', 'intent classification', 'intent classifier', 'classifies intent', 'classify intent', 'intent detection', 'detects intent', 'news classification', 'classifies news', 'article classification', 'classifies articles', 'news categorization', 'categorizes news', 'assigns labels to text', 'assign labels to text', 'labels text', 'label text', 'predicts text labels', 'predicts labels for text', 'topic classification', 'classifies topics', 'topic classifier', 'text labeling', 'labels documents', 'classifies sentences'))
        self.assertEqual(literals["NOTEXTCLASSIFICATION"], ('no text classification', 'does not classify text', 'without text classification', 'no text classifier', 'does not support text classification', 'classification not supported', 'does not classify documents', 'no document classification', 'no spam detection', 'does not detect spam', 'cannot classify text', 'never classifies text', 'no intent classification', 'does not classify intent', 'no news classification', 'does not support classification', 'text classification not supported', 'does not categorize text', 'no text categorization', 'manual labeling only', 'rule based filtering only', 'keyword matching only', 'no automated classification', 'human review only'))
        self.assertEqual(literals["GENERIC_NAMES"], ('classify_text', 'text_classifier', 'predict_text_class', 'classify_document', 'classify_documents', 'categorize_text', 'text_categorizer', 'predict_text_label', 'predict_text_labels', 'classify_sentence', 'classify_sentences', 'detect_spam', 'spam_classifier', 'classify_intent', 'intent_classifier', 'classify_news', 'classify_article', 'classify_topic', 'topic_classifier', 'label_text', 'label_document', 'classify_texts', 'categorize_document', 'get_text_class'))
        self.assertEqual(literals["SKLEARN_CLASSIFIERS"], ('MultinomialNB', 'BernoulliNB', 'ComplementNB', 'LogisticRegression', 'LinearSVC', 'SVC', 'SGDClassifier', 'RandomForestClassifier', 'DecisionTreeClassifier', 'KNeighborsClassifier', 'Perceptron', 'PassiveAggressiveClassifier'))

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from text_classification_claim_audit.audit import audit, claims, evidence, main

GOOD = [
 'import transformers\np = pipeline("text-classification")\np("x")',
 'from transformers import pipeline as p\np(task="text-classification")("x")',
 'import transformers as t\np=t.pipeline("text-classification")\np("x")',
 'from transformers import AutoModelForSequenceClassification as A\nm=A.from_pretrained("x")\nm("x")',
 'import transformers\ntransformers.TFAutoModelForSequenceClassification.from_pretrained("x")("x")',
 'import spacy\nnlp.add_pipe("textcat")\nnlp("x")',
 'import spacy\nnlp.create_pipe("textcat_multilabel")\nnlp.pipe(texts)',
 'from spacy.pipeline import TextCategorizer as TC\nc=TC(nlp)\nc(doc)',
 'from nltk import NaiveBayesClassifier as NB\nc=NB.train(data)\nc.classify(feats)',
 'import nltk\nnltk.NaiveBayesClassifier.train(data).classify(feats)',
 'from sklearn.feature_extraction.text import TfidfVectorizer as V\nfrom sklearn.naive_bayes import MultinomialNB as C\nv=V()\nc=C()\nc.predict(X)',
 'import sklearn\nv=CountVectorizer()\nc=LogisticRegression()\nc.predict_proba(X)',
 'import sklearn\nv=TfidfVectorizer()\nc=Pipeline(steps)\nc.predict(X)',
 'import sklearn\nv=TfidfVectorizer()\nc=MultinomialNB().fit(X,y)\nc.predict(X)',
 'import sklearn\nv=TfidfVectorizer()\nc=MultinomialNB()\nd=c.fit(X,y)\nd.predict(X)',
 'import sklearn\nv=TfidfVectorizer()\nMultinomialNB().fit(X,y).predict(X)',
 'import sklearn\nv=TfidfVectorizer()\nPipeline(steps).fit(X,y).predict_proba(X)',
 'x.classify_text(text)',
 'from tool import classify_text as ct\nct(text)',
]
BAD = [
 'import transformers, spacy, nltk, sklearn',
 'import transformers\np=pipeline("text-classification")',
 'import transformers\np=pipeline("sentiment-analysis")\np("x")',
 'import transformers\np=pipeline("zero-shot-classification")\np("x")',
 'import transformers\nm=AutoModelForSequenceClassification.from_pretrained("x")',
 'import transformers\nm=AutoModelForTokenClassification.from_pretrained("x")\nm(x)',
 'import spacy\nnlp.add_pipe("textcat")',
 'import spacy\nnlp("x")',
 'import spacy\nc=TextCategorizer(nlp)',
 'import nltk\nc=NaiveBayesClassifier.train(data)',
 'import nltk\nc.classify(x)',
 'import nltk\nc=Other.train(x)\nc.classify(x)',
 'import sklearn\nv=TfidfVectorizer()',
 'import sklearn\nc=MultinomialNB()\nc.predict(x)',
 'import sklearn\nv=TfidfVectorizer()\nc=Other()\nc.predict(x)',
 'import sklearn\nv=TfidfVectorizer()\nc=MultinomialNB().fit(X,y)',
 'TextBlob(text).sentiment\nvader.polarity_scores(text)',
 'LatentDirichletAllocation().fit(X)\nNMF().fit(X)\nLDA().fit(X)',
 'import transformers\np=pipeline("ner")\np(text)',
 'import transformers\np=pipeline("summarization")\np(text)',
 'import transformers\np=pipeline("translation")\np(text)',
 'x.extract_keywords(text)', 'x.detect_language(text)',
 'open(path).read()\npandas.read_csv(path)', 'tokenizer(text)',
 'doc.cats', '"x.classify_text(text)"\n# x.classify_text(text)',
 'x.classify_text', 'classify_text(text)', 'def broken(',
 'subprocess.run(["external", "classify", "text"])',
 'from tool import classify_text\ndef classify_text(x): pass\nclassify_text(text)',
 'class X:\n def classify_text(self,x): pass\nx.classify_text(text)',
 'from transformers import pipeline\ndef pipeline(x): pass\npipeline("text-classification")("x")',
 'import sklearn\nclass MultinomialNB: pass\nv=TfidfVectorizer()\nc=MultinomialNB()\nc.predict(x)',
]

class EvidenceTests(unittest.TestCase):
    def test_every_generic_name(self):
        for name in phrases.GENERIC_NAMES:
            with self.subTest(name=name):
                self.assertTrue(evidence('x.' + name + '(text)'))
                self.assertTrue(evidence('from tool import ' + name + ' as f\nf(text)'))
                self.assertFalse(evidence(name + '(text)'))

    def test_every_classifier(self):
        for name in phrases.SKLEARN_CLASSIFIERS:
            with self.subTest(name=name):
                self.assertTrue(evidence('import sklearn\nv=CountVectorizer()\nc=' + name + '()\nc.predict(X)'))

class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def put(self, name, text):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)

    def cli(self, *args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(['audit', str(self.root)] + list(args))
        return code, out.getvalue()

    def test_claim_and_no_source(self):
        self.put('README.md', 'We classify text.')
        code, output = self.cli('--json')
        self.assertEqual(code, 2)
        result = json.loads(output)
        self.assertEqual(result['verdict'], 'FAIL')
        f = result['findings'][0]
        self.assertEqual(f['rule'], 'documented-text-classification-claim-false')
        self.assertEqual(f['severity'], 'HIGH')
        self.assertTrue(f['message'])

    def test_good(self):
        self.put('README.md', 'We classify text.')
        self.put('app.py', 'x.classify_text(text)')
        self.assertEqual(self.cli()[0], 0)
        self.assertEqual(audit(self.root)['verdict'], 'PASS')

    def test_test_only(self):
        self.put('README.md', 'Text classification')
        self.put('tests/test_app.py', 'x.classify_text(text)')
        self.assertEqual(audit(self.root)['findings'][0]['severity'], 'MEDIUM')
        self.assertEqual(audit(self.root)['findings'][0]['rule'], 'documented-text-classification-claim-test-only')
        self.assertEqual(self.cli()[0], 0)
        self.assertEqual(self.cli('--fail-on', 'medium')[0], 2)
        self.assertEqual(self.cli('--fail-on', 'MEDIUM')[0], 2)

    def test_denial(self):
        self.put('README.md', 'Does not classify text')
        self.assertEqual(self.cli()[0], 0)
        self.put('app.py', 'x.classify_text(text)')
        self.assertEqual(self.cli()[0], 2)
        self.assertEqual(audit(self.root)['findings'][0]['rule'], 'documented-no-text-classification-claim-false')

    def test_no_claim(self):
        self.put('README.md', 'A utility.')
        self.put('app.py', 'x.classify_text(text)')
        self.assertEqual(self.cli()[0], 0)

    def test_split_module(self):
        self.put('README.md', 'Text classification')
        self.put('a.py', 'import transformers\np=pipeline("text-classification")')
        self.put('b.py', 'p(text)')
        self.assertEqual(self.cli()[0], 2)

    def test_exclusions(self):
        self.put('README.md', 'Text classification')
        for folder in ('docs', 'fixtures', 'examples'):
            self.put(folder + '/app.py', 'x.classify_text(text)')
        self.assertEqual(self.cli()[0], 2)
        for filename in ('test_a.py', 'a_test.py'):
            self.put(filename, 'x.classify_text(text)')
        self.assertEqual(audit(self.root)['findings'][0]['severity'], 'MEDIUM')

    def test_symlinks(self):
        self.put('README.md', 'Text classification')
        with tempfile.TemporaryDirectory() as other:
            target = Path(other)
            (target / 'app.py').write_text('x.classify_text(text)')
            (self.root / 'link.py').symlink_to(target / 'app.py')
            (self.root / 'linked').symlink_to(target, target_is_directory=True)
            self.assertEqual(self.cli()[0], 2)

    def test_ignore_and_verbose(self):
        self.put('README.md', 'Text classification')
        self.put('src/app.py', 'x.classify_text(text)')
        self.assertEqual(self.cli('--ignore', 'src', '--ignore', 'elsewhere')[0], 2)
        self.assertIn('Production evidence: src/app.py', self.cli('--verbose')[1])

    def test_docs_selection(self):
        self.put('other.md', 'Text classification')
        self.assertEqual(self.cli()[0], 0)
        self.put('nested/docs/guide.md', 'Text classification')
        self.assertEqual(self.cli()[0], 2)

    def test_usage_errors(self):
        for args in ([], ['unknown'], ['audit', '--fail-on', 'low'], ['audit', '--ignore']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as cm:
                main(args)
            self.assertEqual(cm.exception.code, 1)
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(['audit', str(self.root / 'missing')]), 1)

    def test_module_entry_and_version(self):
        result = subprocess.run([sys.executable, '-m', 'text_classification_claim_audit', '--version'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.strip(), '0.1.0')

class MarkdownTests(unittest.TestCase):
    def test_fences(self):
        for fence in ('```', '~~~', '   ```'):
            self.assertEqual(claims(fence + '\ntext classification\n' + fence), (False, False))
        self.assertEqual(claims('`text classification`'), (True, False))

    def test_normalization_and_denial(self):
        self.assertEqual(claims("Doesn't classify text"), (False, True))
        self.assertEqual(claims('TEXT-CLASSIFICATION'), (True, False))
        self.assertEqual(claims('text\nclassification'), (True, False))
        self.assertEqual(claims('text\n\nclassification'), (False, False))
        self.assertEqual(claims('text classifications'), (False, False))
        self.assertEqual(claims('No text classifier. We classify text.'), (True, True))

# Separate test methods make every positive and negative fixture visible in counts.
def route_test(source, expected):
    def run(self):
        self.assertEqual(evidence(source), expected, source)
    return run

for index, source in enumerate(GOOD):
    setattr(EvidenceTests, 'test_good_%02d' % index, route_test(source, True))
for index, source in enumerate(BAD):
    setattr(EvidenceTests, 'test_bad_%02d' % index, route_test(source, False))

setattr(EvidenceTests, "test_cyclic_fit_binding", route_test("import sklearn\nv=CountVectorizer()\nc=MultinomialNB()\nc=c.fit(X,y)\nc.predict(X)", True))

class VocabularyMatchingTests(unittest.TestCase):
    def test_all_positive_phrases(self):
        for phrase in phrases.TEXTCLASSIFICATION:
            with self.subTest(phrase=phrase):
                self.assertEqual(claims(phrase), (True, False))

    def test_all_negative_phrases(self):
        for phrase in phrases.NOTEXTCLASSIFICATION:
            with self.subTest(phrase=phrase):
                self.assertEqual(claims(phrase), (False, True))

setattr(EvidenceTests, 'test_direct_unpaired_classifier', route_test('import sklearn\nv=CountVectorizer()\nMultinomialNB().predict(X)', False))
setattr(EvidenceTests, 'test_unbound_textcat_chain', route_test('import spacy\nTextCategorizer(nlp)(doc)', False))
