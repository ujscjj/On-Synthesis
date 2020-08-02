# On-Synthesis

This is the script for creating the datasets in the paper "On Synthesis for Supervised Monaural Speech Separation in Time Domain", which has been accepted by Interspeech2020.

You just need to excute the following codes:

> ```python
> python create_path.py
> python create_dataset.py
> python create_dataset_increment.py
> 
> tips: the path in ".py" should be assigned according to your storage path
> libri_path = '' # the absolute path for LibriSpeech data, you can download the corpus from 'http://www.openslr.org/12'
> noise_path = '' # the absolute path for noise data, you can download the corpus from 'http://web.cse.ohio-state.edu/pnl/corpus/HuNonspeech/HuCorpus.html'
> ```

then you can get the LS-2mix and LS-2mixNoise dataset in our paper. 



### References

http://www.merl.com/demos/deep-clustering/create-speaker-mixtures.zip

https://github.com/yluo42/TAC/tree/master/data

