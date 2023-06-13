#!/bin/bash
cd lung80
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
