# Translation-Model-with-BiSLTM-and-Transformers
Mô hình với nhiệm vụ dịch máy đơn giản English to Vietnamese

# Systems Requirments
Chúng tôi sử dụng Laptop ASUS G14 ROG ZENPHYRUS với cấu hình như sau:
* CPU: Rezen 7 4000
* GPU: NVIDA GEFORCE GTX 1060 4G
* RAM: 8G
* Hệ điều hành Win 11

Môi trường như sau
* Python: 3.9.18
* Tensorflow-gpu: 2.10.1
* Keras-nlp: 2.10.0
* nltk: 3.8.1
* gradio

# Result
* Trong quá trình này Acc và Loss được tính trong quá trình train model
* Điểm Bleu (đánh giá chất lượng bản dịch) được tính dựa trên khoảng 2000 mẫu bất kỳ trong dữ liệu (Sau quá trình train)
* Thời gian Train với BiLSTM_epoch10 là 57.25p, và Transformers_epoch10 là 35.75p
* Thời gian đánh giá Bleu tương đối lâu vì phải dịch từng bản ghi (Chiếm khoảng 90% thời gian đánh giá)

|            | BiLSTM_epoch10 | Transformers_epoch10 |
|------------|----------------|----------------------|
| Acc_train  | 0.9565         | 0.8271               |
| Loss_train | 0.1836         | 0.2765               |
| Acc_val    | 0.1836         | 0.8033               |
| Loss_val   | nan            | 0.3163               |
| **Bleu**       | **0.5364**         | **0.6105**               |

# Lưu ý
* Vui lòng ghi đường dẫn nếu kho lưu trữ này giúp ích cho bạn 💕💕💕

