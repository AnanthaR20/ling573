#  Deliverable 3 Run Details

Each folder here represents a run/pipeline executed during deliverable 3.
Using BillSum training data, we clean the data, chunk the document, do text simplificiation and finally generate summaries.

The format of the directory name is: 

{Model}\_{Clean}\_{Chunk}\_{Simplify}\_{Training Instance Count}

# Flag Options

{Model} = Model Name

pegasusbillsum : [Pegasus BillSum](https://huggingface.co/google/pegasus-billsum)

led : [longformer encoder decoder]()

---
{Clean} = Binary flag indicating whether to use clean or unclean billsum data. No flag means "unclean"/original data

clean : indicates whether billsum document text is "cleaned" (using regexes to remove Section headers and parentheses, etc)

---
{Chunk} = Method of chunking. Deterministic chunking or Se3 chunking

det : chunking of bill text into fixed size blocks

se3 : chunking of bill text based on semantic chunking done by [se3 model]()

---
{Simplify} = Automatic Text Simplification module to use.

TSR : [Text Split and Rephrase]()

DisSim : [Discourse Simplification](https://github.com/Lambda-3/DiscourseSimplification)

---
{Training Instance Count} = Number of Training Instances used for fine-tuning

\# : The number of training instances ran

toy : means we ran <10 instances

---

If one of these flags is missing, it means the pipeline doesn't use any of the options for the flag as presented above.

