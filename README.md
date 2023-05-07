Download Link: https://assignmentchef.com/product/solved-learning-hypernyms-assignment-8
<br>
NO WRITEUP

In linguistics, <em>Hypernymy</em> is an important lexical-semantic relationship that captures the <em>type-of</em> relation. In this relation, a <strong>hyponym</strong> is a word or phrase whose semantic field is included within that of another word, its <strong>hypernym</strong>. For example, <em>rock</em>, <em>blues</em> and <em>jazz</em> are all hyponyms of <em>music genre</em>(hypernym).

In this assignment, we will examine unsupervised techniques to automatically extract a list of word pairs that satisfy the hypernymy relation using a large corpus. Specifically, we will use

<ol>

 <li>A rule-based technique using lexico-syntactic patterns to extract hyponym-hypernym word pairs.</li>

 <li>Another rule-based technique but using dependency-paths</li>

 <li>A DIY approach where you can use any supervised/unsupervised technique extract hypernym-hyponym word pairs.</li>

</ol>

In this assignment, we will be using <em>nltk</em>. If you are use Stephen’s miniconda you should be fine. Otherwise, install nltk and the relavant tagger using.

<h1 id="part-1-hearst-patterns-for-hypernym-learning">Part 1: Hearst Patterns for Hypernym Learning</h1>

Marti Hearst, in her classic 1992 COLING paper <a href="http://www.aclweb.org/anthology/C92-2082">Automatic Aquisition of Hyponyms from Large Text Cororpa</a>, described how lexico-syntactic patterns can be used for hyponym acquisition.

Consider the sentence,

Upon reading, we can easily infer that <em>blues</em>, <em>rock</em> and <em>jazz</em> are types of <em>musical genres</em>. Consider another sentence that contains a similar lexico-syntactic construction and expresses a hypernymy relation between <em>green vegetables</em> and <em>spinach</em>, <em>peas</em>, and <em>kale</em>.

We can generalize this construction to

where <em>NP_0</em> is the hypernym of <em>NP_1</em>, <em>NP_2</em>, …, <em>NP_n</em>. Here, <em>NP_x</em> refers to a <a href="https://en.wikipedia.org/wiki/Noun_phrase">noun phrase or noun chunk</a>.

As you can see in her paper, Hearst identified many such patterns, hence forth referred to as Hearst Patterns. Since these patterns are already manually constructed, we can use these patterns on a large corpus of unlabeled text for hyponym acquisition. For example, if our corpus only contained the sentences above, we could extract hypo-hypernym pairs such as

In this section of the assignment, we will use these patterns to extract hyper-hyponym word pairs from Wikipedia.

<h3 id="evaluation-dataset">Evaluation Dataset</h3>

To evaluate the quality of our extraction, we will use a post-processed version of the <a href="http://www.aclweb.org/anthology/W11-2501">BLESS2011 dataset</a>. <a href="http://www.aclweb.org/anthology/N15-1098">Levy et al. ‘15</a> post-processed to clean the dataset for reasons mentioned in the paper. Based on our extractions from a large corpus, we can label the test instances as a True hypo-hypernym extraction if it exists in the extracted list or as a False pair if it doesn’t. The data files are tab-separated with one-pair-per-line.

<h3 id="unlabeled-wikipedia-corpus">Unlabeled Wikipedia Corpus</h3>

As our large corpus to extract hyponyms, we will use Wikipedia text. Since Wikipedia is too large for efficient processing, we provide you with a pruned version. This only contains sentences that contain a word pair from train/val/test set. Each line in the file contains 2 tab-separated columns. The first column contains the tokenized sentence, and the second contains the lemmatized version of the same tokenized sentence.

<h2 id="how-to-get-started">How to Get Started</h2>

As you must have noticed, implementing Hearst Patterns requires noun-phrase chunking and then regex pattern matching where these patterns are relevant Hearst patterns.

In the code file <code class="highlighter-rouge">hearst/hearstPatterns.py</code> we use NLTK to implement a <a href="http://www.nltk.org/book/ch07.html">nltk-regex based noun-phrase chunker</a> and Hearst pattern matching.

<ol>

 <li>This code first finds the all the noun-chunks and converts the sentence into the following format,</li>

 <li>It then uses nltk-regex to find patterns defined in the list <code class="highlighter-rouge">self.__hearst_patterns</code>. We have already implemented one pattern (NP_0 such as NP1, …) for you as:which will lead to extractions such as</li>

</ol>

Your job is to implement other Hearst Patterns and add them to the <code class="highlighter-rouge">self.__hearst_patterns</code> list.

To use the method above to perform large-scale extraction on Wikipedia, and evaluation against the provided dataset, we provide the following functions:

<ul>

 <li><code class="highlighter-rouge">hearst/extractHearstHyponyms.py</code> – Implements a method to apply the Hearst Patterns to all Wikipedia sentences, collect the extractions and write them to a file</li>

 <li><code class="highlighter-rouge">extractDatasetPredictions.py</code> – Labels the train/val/test word-pairs as True (False) if they exist (don’t exist) in the extracted hypo-hypernym pairs</li>

 <li><code class="highlighter-rouge">computePRF.py</code> – Takes the gold-truth and prediction file to compute the Precision, Recall and F1 score.</li>

</ul>

You should implement different Hearst Patterns, and/or come up with your own patterns by eyeballing Wikipedia data. Use the train and validation data to estimate the performance of different pattern combinations and submit the predictions from the best model on the test data to the leaderboard as the file <code class="highlighter-rouge">hearst.txt</code>. The format of this file will the same as the train and validation data.

Don’t panic if your validation performance is too low (~30%-36% for lemmatized corpus). Using the single pattern already implemented, you should get a validation score of 30% (lemmatized corpus).

In a <code class="highlighter-rouge">writeup.pdf</code> explain which patterns helped the most, and patterns that you came up with yourself.

<h2 id="hint-for-post-processing-extractions">Hint for post-processing extractions</h2>

As you might notice in the dataset, the (hypernym, hyponym) pair contains single, usually lemmatized, tokens. On the other hand, the extractions from our code extracts a lot of multi-word noun phrases. This could negatively affect our performance on the provided dataset. Our performance could drastically improve if we post-process our extractions to output only single token extractions.

You should be able to use your knowledge about noun-phrases (that the head of the noun-phrase usually is the last token) and lemmatization to figure out a post-processing methodology that might improve your performance. In the final report, explain your post-processing methodology, and make clear distinction between change in performance when the extractions were post-processed.

<h1 id="part-2-dependency-paths-for-hypernym-learning">Part 2: Dependency Paths for Hypernym Learning</h1>

Consider the sentence snippet, <em>… such green vegetables as spinach, peas and kale.</em> and its dependency-parse using <a href="https://spacy.io/">spaCy</a>. Their visualization tool is called displaCy and can be accessed here: <a href="https://demos.explosion.ai/displacy/">demos.explosion.ai/displacy</a>

<img decoding="async" alt="Example dependency path using spaCy" data-recalc-dims="1" data-src="https://i0.wp.com/computational-linguistics-class.org/assets/img/deppath.png?w=980" class="lazyload" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==">

 <noscript>

  <img decoding="async" src="https://i0.wp.com/computational-linguistics-class.org/assets/img/deppath.png?w=980" alt="Example dependency path using spaCy" data-recalc-dims="1">

 </noscript>

We (again) observe that there are dependency paths between <em>vegetables</em> and <em>spinach</em>, <em>peas</em>, and <em>kale</em> that can be typical in expressing hyper-hyponymy relation between words.

This observation was made in Snow et al.’s NIPS 2005 <a href="https://papers.nips.cc/paper/2659-learning-syntactic-patterns-for-automatic-hypernym-discovery.pdf">Learning syntactic patterns for automatic hypernym discovery</a> paper. In this paper, the authors use the shortest dependency paths between noun-phrases to extract features and learn a classifier to predict whether a word-pair satisfies the hypo-hypernymy relation.

On the contrast, in this part, we will use the shortest dependency paths between noun (noun-chunk) pairs to predict hypernymy relation in an unsupervised manner. For example, in the example above the shortest dependency path between <em>vegetables</em> and <em>spinach</em> is

For generalization, we can anonymize the start and end-points of the paths, as

and can now predict that when a (X,Y) pair occurs in such a dependency-path, it usually means that X is the hypernym of Y.

Simply using the shortest dependency paths as two major issues that can be tackled in the following manner:

<ul>

 <li><strong>Better lexico-syntactic paths</strong>: Two words containing <em>as</em> in between is not a good indicator of the hypernymy relation being expressed and the word <em>such</em> outside the shortest path plays an important role. Snow et al. hence proposed <em>Satellite edges</em>, i.e. adding single edges to the left (right) of the leftmost (rightmost) token to the shortest-dependency path. In the example above, in addition to the shortest-path, the paths we extract will also contain satellite edges from <em>vegetable -&gt; green</em>, <em>vegetable -&gt; such</em>, and <em>spinach -&gt; peas</em>.</li>

 <li><strong>Distributive Edges</strong>: The shortest path from <em>vegetables</em> from <em>peas</em> contains <em>spinach/NOUN</em> node connected via the <em>conj</em> edge. The presence of such specific words (<em>spinach</em>) in the path that do not inform of the hypernymy relation and negatively affect our extraction recall. The word <em>spinach</em>could be replaced with some other word, but the relation between peas and vegetables would still hold. Snow et al. proposed to add additional edges bypassing <em>conj</em> edges to mitigate this issue. Therefore, we can add edges of type <em>pobj</em> from <em>vegetables</em> to <em>peas</em> and <em>kale</em>.</li>

</ul>

<strong>Good News</strong>: You don’t need to extract such paths! We’ve extracted them for you for our Wikipedia corpus. You’re welcome! To keep the number of extracted paths tractable, the file <code class="highlighter-rouge">wikipedia_deppaths.txt</code> contains dependency paths extracted from Wikipedia between all train/val/test word pairs.Similar to Snow et al., we <strong>only keep dependency paths of length 4 or shorter</strong> . Each line of the file <code class="highlighter-rouge">wikipedia_deppaths.txt</code> contains three tab-separated columns, the first containing <strong>X</strong>, second <strong>Y</strong> and the third the dependency path. An example path extraction is:

The path edges are delimited by the underscore ( <em>_</em> ). Each edge contains <code class="highlighter-rouge">word/POS_TAG/EdgeLabel</code>. Hence the edges in the path are:

<ol>

 <li><code class="highlighter-rouge">such/ADJ/amod&lt;</code>: An edge of type <em>amod</em> from the right side (<em>X</em>) to the word <em>such</em></li>

 <li><code class="highlighter-rouge">X/NOUN/dobj</code>: An edge of type <em>dobj</em> from outside the shortest path span (since no direction marker (‘&lt;’ or ‘&gt;’) exists) to the token <em>X</em></li>

 <li><code class="highlighter-rouge">&lt;as/ADP/prep</code>: An edge of type <em>prep</em> from <em>as</em> to the left of it (<em>X</em>).</li>

 <li><code class="highlighter-rouge">&lt;Y/NOUN/pobj</code>: An edge of type <em>pobj</em> from <em>Y</em> to the left of it (<em>as</em>)</li>

</ol>

<h3 id="how-to-proceed">How to proceed</h3>

Similar to how we used the Hearst Patterns to extract hypo-hypernym pairs, we will now use dependency-path patterns to do the same. Since it is difficult to come up with your own dependency path patterns, we suggest you use the the labeled training data to come up with a list of dependency-paths that are positive examples of paths between actual hyper-hyponym pairs.

<ul>

 <li><code class="highlighter-rouge">depPath/extractRelevantDepPaths.py</code>: Fill this python script, to extract relevant dependency paths from <code class="highlighter-rouge">wikipedia_deppaths.txt</code> using the training data and store them to a file Note that, these paths can be of different categories. For eg. Forward paths: Classify X/Y as Hyponym/Hypernym, Reverse paths: Classify X/Y as Hypernym/Hyponym, Negative paths: Classify X/Y as a negative pair etc.</li>

 <li><code class="highlighter-rouge">depPath/extractDepPathHyponyms.py</code>: Complete this python script to generate a list of hypo-hypernym extractions for the wikipedia corpus (<code class="highlighter-rouge">wikipedia_deppaths.txt</code>) (similar to Hearst Patterns)</li>

 <li><code class="highlighter-rouge">extractDatasetPredictions.py</code>: Similar to Hearst patterns, use this to label the train/val/test word-pairs as True (False) if they exist (don’t exist) in the extracted hypo-hypernym pairs.</li>

 <li><code class="highlighter-rouge">computePRF.py</code>: Similar to Hearst Patterns, use this script to evaluate the performance on the given dataset</li>

</ul>

For the best performing dependency paths, submit the test predictions on the relevant leaderboard with the filename <code class="highlighter-rouge">deppath.txt</code>. In the <code class="highlighter-rouge">writeup.pdf</code>explain few most occurring dependency paths and explain if they bear any correspondence to the Hearst patterns.

<h1 id="part-3-diy">Part 3: DIY</h1>



<center>

 <em>Cause I’m as free as a bird now, And this bird you can not change. – Lynyrd Skynyrd</em>

</center>In the previous two sections we saw how we can use manually extracted rule-based techniques to extract word-pairs satisfying hypernymy relations. In this section, you have to implement <strong>at least two additional methods</strong> to extract such word pairs.



A few of example ideas for additional techniques are:

<ul>

 <li>Combine the extractions from Hearst patterns and Dependency-path patterns for better extractions</li>

 <li>Learn a supervised classifier (similar to Snow et al) using the provided training data and features extracted from the dependency paths</li>

 <li>Use pre-trained word embeddings to learn a hypernymy prediction classifier, or combine word-embeddings are features to your dependency–path based classifier.</li>

</ul>

For this part, the test data predictions from the best performing technique should be uploaded to the relevant leaderboard as <code class="highlighter-rouge">diy.txt</code>. In the <code class="highlighter-rouge">writeup.pdf</code> explain in detail the two methodologies implemented with performance analysis.

<h3 id="3-the-leaderboard">3. The Leaderboard</h3>

We will have three leaderboards for this assignment, namely

<ol>

 <li>Hearst Patterns – <code class="highlighter-rouge">hearst.txt</code></li>

 <li>DepdencyPath Patterns – <code class="highlighter-rouge">deppath.txt</code></li>

 <li>DIY Model – <code class="highlighter-rouge">diy.txt</code></li>

</ol>

<h3 id="extra-credit">Extra Credit</h3>

<ul>

 <li>Extra credit to the top-5 teams on each leaderboard.</li>

 <li>Extra credit to teams that improve their best performing model from Part 1/2 by 5% in their DIY model.</li>

</ul>

<h2 id="deliverables">Deliverables</h2>

<h2 id="recommended-readings">Recommended readings</h2>

<table>

 <tbody>

  <tr>

   <td><a href="https://web.stanford.edu/~jurafsky/slp3/21.pdf">Relation Extraction (Section 21.2)</a> Dan Jurafsky and James H. Martin. Speech and Language Processing (3rd edition draft) .</td>

  </tr>

  <tr>

   <td><a href="http://papers.nips.cc/paper/2659-learning-syntactic-patterns-for-automatic-hypernym-discovery.pdf">Learning syntactic patterns for automatic hypernym discovery</a> Rion Snow, Daniel Jurafsky, Andrew Y. Ng. NIPS 2003.</td>

  </tr>

  <tr>

   <td><a href="http://www.aclweb.org/anthology/C92-2082">Automatic acquisition of hyponyms from large text corpora</a> Marti Hearst. COLING 1992.</td>

  </tr>

  <tr>

   <td><a href="http://www.aclweb.org/anthology/N15-1098">Do Supervised Distributional Methods Really Learn Lexical Inference Relations?</a> Omer Levy, Steffen Remus, Chris Beimann, Ido Dagan. NAACL 2015.</td>

  </tr>

  <tr>

   <td><a href="https://arxiv.org/abs/1603.06076">Improving Hypernymy Detection with an Integrated Path-based and Distributional Method</a> Vered Shwartz, Yoav Goldberg and Ido Dagan. ACL 2016.</td>

  </tr>

 </tbody>

</table>

5/5 - (3 votes)

<pre class="highlight"><code>pip install nltk$ python&gt;&gt;&gt; import nltk&gt;&gt;&gt; nltk.download('punkt')&gt;&gt;&gt; nltk.download('averaged_perceptron_tagger')</code></pre>

Here are the materials that you should download for this assignment:

<ul>

 <li><a href="http://computational-linguistics-class.org/downloads/hw8/lexicalinference.zip"><code class="highlighter-rouge">lexicalinference.zip</code></a> Contains all the relevant code.</li>

 <li><a href="https://drive.google.com/drive/folders/1EBZs5L2rbF0immOetBTC3AZbgf4XYtMG?usp=sharing"><code class="highlighter-rouge">bless2011/</code></a> Contains the train, validation and test data</li>

 <li><a href="https://drive.google.com/drive/folders/1EBZs5L2rbF0immOetBTC3AZbgf4XYtMG?usp=sharing"><code class="highlighter-rouge">wikipedia_sentences.txt</code></a> Contains tokenized relevant wikipedia sentences. A lemmatized version also exists.</li>

 <li><a href="https://drive.google.com/drive/folders/1EBZs5L2rbF0immOetBTC3AZbgf4XYtMG?usp=sharing"><code class="highlighter-rouge">wikipedia_deppaths.txt</code></a> Contains word pairs and the shortest dependency path between them as extracted using spaCy</li>

</ul>

Alternatively, if you are on <code class="highlighter-rouge">biglab</code>, you can directly copy the last three resources into your working directory by using the following command:

<pre class="highlight"><code>scp -r /home1/n/nitishg/hw8resources ./</code></pre>

<pre class="highlight"><code>How do I distinguish among different musical genres such as blues, rock, and jazz, etc., and is there a good listeners’ trick to discern such distinctions?</code></pre>

<pre class="highlight"><code>I am going to get green vegetables such as spinach, peas and kale.</code></pre>

<pre class="highlight"><code>NP_0 such as NP_1, NP_2, ... (and | or) NP_n</code></pre>

<pre class="highlight"><code>(blues, musical genres), (rock, musical genres), (jazz, musical genres), (spinach, green vegetables), (peas, green vegetables), (kale, green vegetables)</code></pre>

<pre class="highlight"><code>hyponym t hypernym t label</code></pre>

<pre class="highlight"><code>I like to listen to NP_music from NP_musical_genres such as NP_blues , NP_rock and NP_jazz .</code></pre>

<pre class="highlight"><code>"(NP_w+ (, )?such as (NP_w+ ? (, )?(and |or )?)+)", "first"</code></pre>

<pre class="highlight"><code>('blues', 'musical genres'), ('rock', 'musical genres'), and ('jazz', 'musical genres')</code></pre>

<pre class="highlight"><code>hyponym t hypernym t True(False)</code></pre>

<pre class="highlight"><code>vegetables/NOUN -&gt; Prep -&gt; as/ADP -&gt; pobj -&gt; spinach/NOUN</code></pre>

<pre class="highlight"><code>X/NOUN -&gt; Prep -&gt; as/ADP -&gt; pobj -&gt; Y/NOUN</code></pre>

<pre class="highlight"><code>mammal  fox     such/ADJ/amod&lt;_X/NOUN/dobj_&lt;as/ADP/prep_&lt;Y/NOUN/pobj</code></pre>

Here are the deliverables that you will need to submit. The writeup needs to be submit as <code class="highlighter-rouge">writeup.pdf</code>:

<ul>

 <li>The three prediction files, <code class="highlighter-rouge">hearst.txt</code>, <code class="highlighter-rouge">deppath.txt</code> and <code class="highlighter-rouge">diy.txt</code></li>

 <li>A <code class="highlighter-rouge">writeup.pdf</code> containing

  <ul>

   <li>Written analysis on additional Hearst Patterns implemented and which one worked the best/worst</li>

   <li>Written analysis commenting on the Precision/Recall values when using Hearst Patterns</li>

   <li>Written analysis on most frequent Dependency Paths that worked the best</li>

   <li>Written analysis commenting on the Precision/Recall values when using Dependency Paths</li>

   <li>Implementation details of the DIY models and performance analysis</li>

  </ul></li>

 <li>Your code (.zip) with a <code class="highlighter-rouge">README</code> to run. It should be written in Python 3.</li>