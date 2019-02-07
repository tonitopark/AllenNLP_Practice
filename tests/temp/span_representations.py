# To use span representations need to consider 3 things:

# 1. enumerating all possible spans in ta DatasetReader as input to your model
# 2. extracting embedded span representations for the span indices
# 3. pruning the spans in your model to keep the most promising ones


# Generating SpanFields from text in a DtasetReader



# Extracting span representations form a text sequence

# SpanExtractor --> takes a sequence of tensor of shape :
#                   (batch_size,sentence_length,embedding_size)
#                   and some indices (batch_size,num_spans,2)

#               --> returns encoded representation of each span
#                   of shape ( batch_size,num_spans,encoded_size)

# EndpointSpanExtractor (Simplest one)
#  --> spans as a combination of the embeddings of their endpoints

import torch
from torch.autograd import Variable
from allennlp.modules.span_extractors import EndpointSpanExtractor, SelfAttentiveSpanExtractor

sequence_tensor = Variable(torch.randn([2,5,7]))
print(sequence_tensor)

# concatenate start and end points together to form our representation.
extractor = EndpointSpanExtractor(input_dim=7, combination="x,y,x+y")
sa_extractor = SelfAttentiveSpanExtractor(input_dim=7)
#Typically these would come from a SpanField,
#rather than beding created directly,
indices = Variable(torch.LongTensor([[[1,1],
                                      [2,4]],
                                     [[0,2],
                                      [3,4]]]))
print(indices.size())


# We concatenated the representations for the start and end of the span,
# so the embedded span size is 2 * embedded_size.
# Sahpe (batch_size, num_spans, 2 * embedding_size )

span_representations = extractor(sequence_tensor,indices)
sa_span_representations = sa_extractor(sequence_tensor,indices)
print(list(sa_span_representations))
assert list(span_representations.size()) == [2,2,21]

# Scoring and Pruning Spans

   # Span-based representation are effective for structured prediction problems
   # Drawback  -> span enumeration ( consider all spans in document : n^2)
   #           -> when mixed with co-reference model in AllenNLP --> n^4


# Method 1 : Heuristically prune spans in your DatasetReader

  # included utility function that excludes some condition based on
  # the input text or an Spacy Token attribute
  # for co-reference example, no spans start or end with punctuation


from typing import List
from allennlp.data.dataset_readers.dataset_utils import span_utils
from allennlp.data.tokenizers.word_tokenizer import SpacyWordSplitter
from allennlp.data.tokenizers.token import Token

tokenizer = SpacyWordSplitter(pos_tags=True)
sentence = tokenizer.split_words("This is a sentence.")

def no_prefixed_punctuation(tokens: List[Token])->bool:
    # Only include spans which don't start or end with punctuation.
    return tokens[0].pos_ !="PUNCT" and tokens[-1].pos_ !="PUNCT"

spans = span_utils.enumerate_spans(sentence,
                                   max_span_width=3,
                                   min_span_width=2,
                                   filter_function=no_prefixed_punctuation)
spans_no_filter = span_utils.enumerate_spans(sentence,
                                   max_span_width=3,
                                   min_span_width=2)

# 'spans' won't include (2,4) or (3,4) as these ahve
# punctuation as their last element. Note that these spans
# have inclusive start and end indices!


print(spans)
print(spans_no_filter)
assert spans ==[(0,1),(0,2),(1,2),(1,3),(2,3)]


  # Method 2 : Use Pruner
      # It is not always possible to prune spans before they enter the model
      # --> provides a Pruner that allows to prune spans based on a parameterized function
      #      which is trained end-to_end with the rest of the model

import torch
from torch.autograd import Variable
from allennlp.modules import SpanPruner

# Create a linear layer which will score our spans.
linear_scorer = torch.nn.Linear(5,1)
pruner = SpanPruner(scorer = linear_scorer)

# randomly fabricate spans output ( as output of SpanExtractor)
# shape (batch_size, num_spans, embedding_size)

spans = Variable(torch.randn([3,4,5]))
mask  = Variable(torch.ones([3,4]))

pruned_embeddings, pruned_mask, pruned_indices, pruned_scores \
= pruner(spans, mask ,num_spans_to_keep=3)

print(pruned_embeddings)
print(pruned_mask)
print(pruned_indices)
print(pruned_scores)

#return values
# pruned_embeddings
     # top-k (num_spans_to_keep) spans selected using loss function

# pruned_mask
     #

# pruned_indices
     # indices of the top-k scoring spans in the original spans tensor

# pruned_scores
     # scores for loss function


