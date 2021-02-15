mkdir data/train
cd data/train
wget -c https://ndownloader.figshare.com/files/25953977 
mv 25953977 tr_im.nii.gz
wget -c https://ndownloader.figshare.com/files/25953980
mv 25953980 tr_mask.nii.gz
cd ..
wget -c https://ndownloader.figshare.com/files/25953974
mv 25953974 Test-Images-Clinical-Details.csv
mkdir test
cd test
wget -c https://ndownloader.figshare.com/files/25953983
mv 25953983 val_im.nii.gz