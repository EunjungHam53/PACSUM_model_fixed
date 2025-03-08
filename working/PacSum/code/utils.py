from rouge_score import rouge_scorer
import os, shutil, random, string

from gensim_preprocess import preprocess_documents


def evaluate_rouge(summaries, references, remove_temp=False):
    '''
    Args:
        summaries: [[sentence]]. Each summary is a list of strings (sentences)
        references: [[[sentence]]]. Each reference is a list of candidate summaries.
        remove_temp: bool. Whether to remove the temporary files created during evaluation.
    '''
    assert len(summaries) == len(references)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    results = []
    for summary, candidates in zip(summaries, references):
        best_score = None
        for candidate in candidates:
            candidate_text = "\n".join(candidate)
            summary_text = "\n".join(summary)
            scores = scorer.score(candidate_text, summary_text)
            
            # Chỉ lưu điểm cao nhất trong số các bản tóm tắt tham chiếu
            if best_score is None or scores['rouge1'].fmeasure > best_score['rouge1'].fmeasure:
                best_score = scores
        
        results.append(best_score)

    return results


def clean_text_by_sentences(text):
    """Tokenize a given text into sentences, applying filters and lemmatize them.

    Parameters
    ----------
    text : str
        Given text.

    Returns
    -------
    list of :class:`~gensim.summarization.syntactic_unit.SyntacticUnit`
        Sentences of the given text.

    """
    original_sentences = text
    filtered_sentences = [join_words(sentence) for sentence in preprocess_documents(original_sentences)]

    return filtered_sentences


def join_words(words, separator=" "):
    """Concatenates `words` with `separator` between elements.

    Parameters
    ----------
    words : list of str
        Given words.
    separator : str, optional
        The separator between elements.

    Returns
    -------
    str
        String of merged words with separator between elements.

    """
    return separator.join(words)
