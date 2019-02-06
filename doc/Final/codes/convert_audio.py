#bin/import_librivox
for root, dirnames, filenames in os.walk(source_dir):
    for filename in fnmatch.filter(filenames, '*.trans.txt'):
        trans_filename = os.path.join(root, filename)
        with codecs.open(trans_filename, "r", "utf-8") as fin:
            for line in fin:
                # Parse each segment line
                first_space = line.find(" ")
                seqid, transcript = line[:first_space], line[first_space + 1:]

                # We need to do the encode-decode dance here because encode
                # returns a bytes() object on Python 3, and text_to_char_array
                # expects a string.
                transcript = unicodedata.normalize("NFKD", transcript) \
                    .encode("ascii", "ignore") \
                    .decode("ascii", "ignore")

                transcript = transcript.lower().strip()

                # Convert corresponding FLAC to a WAV
                flac_file = os.path.join(root, seqid + ".flac")
                wav_file = os.path.join(target_dir, seqid + ".wav")
                if not os.path.exists(wav_file):
                    Transformer().build(flac_file, wav_file)
                wav_filesize = os.path.getsize(wav_file)

                files.append((os.path.abspath(wav_file), wav_filesize, transcript))

return pandas.DataFrame(data=files, columns=["wav_filename", "wav_filesize", "transcript"])