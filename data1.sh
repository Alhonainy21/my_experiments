#!/bin/bash
cd /
curl -L -o sss https://www.dropbox.com/s/e189wlk7kkdlhgs/lung_80.tar.gz?dl=0
tar -zxvf sss
mv lung80,20 lung80
cd lung80
find -name '._*' -delete
cd test
cd lung_aca
cd acatest1
mv *jpeg /lung80/test/lung_aca
cd ..
rm -r aca*
cd ..
cd lung_n
cd ntest1
mv *jpeg /lung80/test/lung_n
cd ..
rm -r nte*
cd ..
cd lung_scc
cd stest1
mv *jpeg /lung/test/lung_scc
cd ..
rm -r ste*
cd ../..
cd train
cd lung_aca
cd acatrain1
mv *jpeg /lung80/train/lung_aca
cd ..
rm -r aca*
cd ..
cd lung_n
cd ntrain1
mv *jpeg /lung80/train/lung_n
cd ..
rm -r ntr*
cd ..
cd lung_scc
cd strain1
mv *jpeg /lung/train/lung_scc
cd ..
rm -r str*
cd ../../..
