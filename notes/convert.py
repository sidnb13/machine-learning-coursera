#!/usr/local/bin/python3

import os, sys
from colorama import Fore

if __name__ == "__main__":

    files = [str(file) for file in os.listdir('./src/')]
    files.sort()

    def convert_pdf(md_file):
        fileName = str(md_file).replace('.md','')
        os.system(f'pandoc ./src/{md_file} --pdf-engine=xelatex -o ./pdf/{fileName}.pdf -V geometry:margin=1in')

    if len(sys.argv) > 1:
        pageRanges = sys.argv[1].split(',')
        
        for token in pageRanges:
            pages = []
            if '-' in token:
                pages = range(int(token.split('-')[0]) - 1,int(token.split('-')[1]))
            else:
                pages = range(int(token) - 1,int(token))


            for i in pages:
                if i >= len(files):
                    print(Fore.RED + 'Enter valid page range')
                    exit(1)

                print(f'[{pages.index(i) + 1} of {len(pages)}] Converting {files[i]}')
                convert_pdf(files[i])
                print('Done')
        