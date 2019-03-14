cmd_orders
==========
# cmd_orders local
### First of all-file encoding
```
 In the first, you should see the encoding of the file 'im2latex_formulas.lst'. eapically process the labels 1.Open this file in vim and in the 'esc' model
 2.Type the cmd ':set fileencoding' then you can see  the encoding of this file.
 3.Make this file encoding 'utf-8': type the cmd ': set fileencoding = utf-8'
 4.Check the change: again type the cmd:':set fileencoding' and see the file encoding.

```

### images
* 进行png文件的剪裁，剪裁得到最小矩形边框
```
cd process
python3 scripts/preprocessing/preprocess_images.py --input-dir /home/xiaofeng/data/char_formula/img_ori --output-dir /home/xiaofeng/data/char_formula/prepared/img
```
* 对公式进行最小边框裁剪
```
python3 crop_image.py 
```
* 对增强的数据进行图片剪裁-remote
```
cd process
python3 scripts/preprocessing/preprocess_images.py --input-dir /home/xiaofeng/data/char_formula/img_enhance --output-dir /home/xiaofeng/data/char_formula/prepared_enhance/img
```

### labels
对label进行标准化，分割字符，插入空格-纯公式
```
python3 scripts/preprocessing/preprocess_formulas.py --mode normalize --input-file /Users/xiaofeng/Code/graphica/Chinese_latex_OCR/dataset/formula/dataset/temp/formula.lst  --output-file /Users/xiaofeng/Code/graphica/Chinese_latex_OCR/dataset/formula/dataset/temp/formula_normal_ori.lst
```

```
对于文本+公式结构split_char_formula.py将文本+公式结构进行标准化
```

### full dataset
* 对整个数据集进行处理获得词表-本地路径

```
python3 scripts/preprocessing/preprocess_filter.py --filter --image-dir /Users/xiaofeng/Code/Github/dataset/charactor_formula/prepared/img --label-path /Users/xiaofeng/Code/Github/graphic/Character_mathjax_ocr/dataset/char_formula/new_char_formula_normal_enhance_out.ls  --data-path /Users/xiaofeng/Code/Github/graphic/Character_mathjax_ocr/dataset/char_formula/dataset_char_formula_enhance.ls  --output-path /Users/xiaofeng/Code/Github/graphic/Character_mathjax_ocr/dataset/char_formula/dataset_char_formula_enhance_filter.ls
* 从整个数据集中获得词表
python3 scripts/preprocessing/generate_latex_vocab.py --data-path /Users/xiaofeng/Code/Github/graphic/Character_mathjax_ocr/dataset/char_formula/dataset_char_formula_enhance_filter.ls --label-path /Users/xiaofeng/Code/Github/graphic/Character_mathjax_ocr/dataset/char_formula/new_char_formula_normal_enhance_out.ls  --output-file /Users/xiaofeng/Code/Github/graphic/Character_mathjax_ocr/dataset/char_formula/char_formula_full_enhance_db.txt
```
* 对整个数据集进行处理获得词表-集群路径

```
python3 scripts/preprocessing/preprocess_filter.py --filter --image-dir /home/xiaofeng/data/char_formula/prepared_enhance/img --label-path /home/xiaofeng/code/Character_mathjax_ocr/dataset/char_formula/new_char_formula_normal_enhance_out.ls  --data-path /home/xiaofeng/code/Character_mathjax_ocr/dataset/char_formula/dataset_char_formula_enhance.ls  --output-path /home/xiaofeng/code/Character_mathjax_ocr/dataset/char_formula/dataset_char_formula_enhance_filter.ls
* 从整个数据集中获得词表
python3 scripts/preprocessing/generate_latex_vocab.py --data-path /Users/xiaofeng/Code/graphica/Chinese_latex_OCR/dataset/formula/dataset/data_label/dataset.lst --label-path /Users/xiaofeng/Code/graphica/Chinese_latex_OCR/dataset/formula/dataset/data_label/formula_normal.lst --output-file /Users/xiaofeng/Code/graphica/Chinese_latex_OCR/dataset/formula/dataset/data_label/char_vocab.txt
```

### train.filter
```
纯公式路径
python3 scripts/preprocessing/preprocess_filter.py --filter --image-dir /Users/xiaofeng/Code/Github/dataset/formula/generate/images_processed --label-path /Users/xiaofeng/Code/Github/dataset/formula/generate/prepared/formulas.norm.lst --data-path /Users/xiaofeng/Code/Github/im2latex-tensorflow/im2latex-dataset/generate/train.list --output-path /Users/xiaofeng/Code/Github/dataset/formula/generate/prepared/train_filter.lst
文本+公式
python3 scripts/preprocessing/preprocess_filter.py --filter --image-dir /home/xiaofeng/data/char_formula/prepared/img --label-path /home/xiaofeng/code/Character_mathjax_ocr/dataset/char_formula/new_char_formula_normal.ls  --data-path /home/xiaofeng/code/Character_mathjax_ocr/dataset/char_formula/dataset_char_formula_train.ls --output-path /home/xiaofeng/code/Character_mathjax_ocr/dataset/char_formula/dataset_char_formula_train_filter.ls
文本+公式--增强
python3 scripts/preprocessing/preprocess_filter.py --filter --image-dir /home/xiaofeng/data/char_formula/prepared_enhance/img --label-path /home/xiaofeng/code/Character_mathjax_ocr/dataset/char_formula/new_char_formula_normal_enhance_out.ls  --data-path /home/xiaofeng/code/Character_mathjax_ocr/dataset/char_formula/dataset_char_formula_enhance_train.ls --output-path /home/xiaofeng/code/Character_mathjax_ocr/dataset/char_formula/dataset_char_formula_enhance_train_filter.ls
```

### validate.filter
```
python3 scripts/preprocessing/preprocess_filter.py --filter --image-dir /Users/xiaofeng/Code/Github/dataset/formula/generate/images_processed --label-path /Users/xiaofeng/Code/Github/dataset/formula/generate/prepared/formulas.norm.lst --data-path /Users/xiaofeng/Code/Github/im2latex-tensorflow/im2latex-dataset/generate/validate.list --output-path /Users/xiaofeng/Code/Github/dataset/formula/generate/prepared/validate_filter.lst
文本+公式
python3 scripts/preprocessing/preprocess_filter.py --filter --image-dir /home/xiaofeng/data/char_formula/prepared/img --label-path /home/xiaofeng/code/Character_mathjax_ocr/dataset/char_formula/new_char_formula_normal.ls  --data-path /home/xiaofeng/code/Character_mathjax_ocr/dataset/char_formula/dataset_char_formula_validate.ls --output-path /home/xiaofeng/code/Character_mathjax_ocr/dataset/char_formula/dataset_char_formula_validate_filter.ls
文本+公式--增强
python3 scripts/preprocessing/preprocess_filter.py --filter --image-dir /home/xiaofeng/data/char_formula/prepared_enhance/img --label-path /home/xiaofeng/code/Character_mathjax_ocr/dataset/char_formula/new_char_formula_normal_enhance_out.ls  --data-path /home/xiaofeng/code/Character_mathjax_ocr/dataset/char_formula/dataset_char_formula_enhance_validate.ls --output-path /home/xiaofeng/code/Character_mathjax_ocr/dataset/char_formula/dataset_char_formula_enhance_validate_filter.ls
```
### vocabulary
```
python3 scripts/preprocessing/generate_latex_vocab.py --data-path /Users/xiaofeng/Code/Github/dataset/formula/generate/prepared/train_filter.lst --label-path /Users/xiaofeng/Code/Github/dataset/formula/generate/prepared/formulas.norm.lst --output-file /Users/xiaofeng/Code/Github/dataset/formula/generate/prepared/latex_vocab.txt
文本+公式
python3 scripts/preprocessing/generate_latex_vocab.py --data-path /home/xiaofeng/code/Character_mathjax_ocr/dataset/char_formula/dataset_char_formula_train_filter.ls --label-path /home/xiaofeng/code/Character_mathjax_ocr/dataset/char_formula/new_char_formula_normal.ls --output-file /home/xiaofeng/code/Character_mathjax_ocr/dataset/char_formula/char_formula.txt
```

# cmd in remote enhance

### **images**
```
cd im2markup
python3 scripts/preprocessing/preprocess_images.py --input-dir /home/xiaofeng/data/formula/generate_enhance/ori/img_ori --output-dir /home/xiaofeng/data/formula/generate_enhance/prepared/images_processed
```

### **labels**

```
python3 scripts/preprocessing/preprocess_formulas.py --mode normalize --input-file /home/xiaofeng/data/formula/generate_enhance/ori/im2latex_formulas_enhance.lst --output-file /home/xiaofeng/data/formula/generate_enhance/prepared/formulas.norm.lst

move from: /home/xiaofeng/data/formula/generate_enhance/ori/im2latex_formulas_enhance.lst
       to:  /home/xiaofeng/data/formula/generate_enhance/prepared/formulas.norm.lst
and rename.
```

### **train.filter**
```
python3 scripts/preprocessing/preprocess_filter.py --filter --image-dir /home/xiaofeng/data/formula/generate_enhance/prepared/images_processed --label-path /home/xiaofeng/data/formula/generate_enhance/prepared/formulas.norm.lst --data-path /home/xiaofeng/data/formula/generate_enhance/ori/train_enhance.list --output-path /home/xiaofeng/data/formula/generate_enhance/prepared/train_filter.lst
```

### **validate.filter**
```
python3 scripts/preprocessing/preprocess_filter.py --filter --image-dir /home/xiaofeng/data/formula/generate_enhance/prepared/images_processed --label-path /home/xiaofeng/data/formula/generate_enhance/prepared/formulas.norm.lst --data-path /home/xiaofeng/data/formula/generate_enhance/ori/validate_enhance.list --output-path /home/xiaofeng/data/formula/generate_enhance/prepared/validate_filter.lst
```

### **vocabulary**
```
python3 scripts/preprocessing/generate_latex_vocab.py --data-path /Users/xiaofeng/Code/Github/dataset/charactor_formula/original_data/prepared/train_filter.lst --label-path /Users/xiaofeng/Code/Github/dataset/charactor_formula/original_data/prepared/formulas.norm.lst --output-file /Users/xiaofeng/Code/Github/dataset/charactor_formula/original_data/prepared/latex_vocab.txt



```





