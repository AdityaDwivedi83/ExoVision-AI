from lightkurve import search_targetpixelfile
print("Searching NASA MAST...")

search_result = search_targetpixelfile("Kepler-10", mission = "Kepler")
print (search_result)

print ("\nDownloading Target Pixel File...")

tpf = search_result.download()

print("\nDownload complete!!")

print(tpf)
print(type(tpf))