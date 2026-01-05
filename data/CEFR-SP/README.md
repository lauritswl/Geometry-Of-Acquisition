# CEFR-Based Sentence Difficulty Annotation and Assessment

CEFR-SP provides 17k English sentences annotated with CEFR-levels assigned by English-education professionals.

It consists of sentences sampled from [Wiki-Auto](https://github.com/chaojiang06/wiki-auto), and [SCoRE](https://www.score-corpus.org/en/).  
We separate train/validation/test sets according to the data-sources. 
Each subset is distributed under the same license of the original corpus.  

## License
* Wiki-Auto: CC BY-SA 3.0
* SCoRE: CC BY-NC-SA 4.0 


## Format
Each directory contains train, validation (dev), and test set files, which is a tab delimited format.

```
Sentence \t Label by annotator A \t Label by annotator B
```

The CEFR-levels are converted to numerals: "1" indicates A1, "2" indicates A2, to "6" indicates C2. 

## Citation
Please cite the following paper if you use the above resources for your research.
 ```
 Yuki Arase, Satoru Uchida, and Tomoyuki Kajiwara. 2022. CEFR-Based Sentence-Difficulty Annotation and Assessment. 
 in Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP 2022) (Dec. 2022).
 
@inproceedings{arase:emnlp2022,
    title = "{CEFR}-Based Sentence-Difficulty Annotation and Assessment",
    author = "Arase, Yuki  and Uchida, Satoru, and Kajiwara, Tomoyuki",
    booktitle = "Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP 2022)",
    month = dec,
    year = "2022",
}
 ```

## Contact
Yuki Arase (arase [at] ist.osaka-u.ac.jp) 
-- please replace " [at] " with an "@" symbol.



