# Load files and perform flood correction.
LoadNexusProcessed(Filename=r'C:/Users/npi34092/Dropbox/Offspec_reduction\GC_Flood_2-19.nxs', OutputWorkspace='wsf1b_flood')
ConvertUnits(InputWorkspace='wsf1b_flood', OutputWorkspace='wsf1b_flood', Target='Wavelength', AlignBins=True)
LoadISISNexus(Filename=r'\\isis.cclrc.ac.uk\inst$\ndxoffspec\instrument\data\cycle_18_4\OFFSPEC00050612.nxs', OutputWorkspace='wtemp')
CloneWorkspace(InputWorkspace='wtemp', OutputWorkspace='w55')
LoadISISNexus(Filename=r'\\isis.cclrc.ac.uk\inst$\ndxoffspec\instrument\data\cycle_18_4\OFFSPEC00050613.nxs', OutputWorkspace='wtemp')
CloneWorkspace(InputWorkspace='wtemp', OutputWorkspace='w56')
ConvertUnits(InputWorkspace='w55', OutputWorkspace='w55', Target='Wavelength', AlignBins=True)
CropWorkspace(InputWorkspace='w55', OutputWorkspace='w55det', StartWorkspaceIndex=5, EndWorkspaceIndex=771)
RebinToWorkspace(WorkspaceToRebin='wsf1b_flood', WorkspaceToMatch='w55det', OutputWorkspace='wsf1b_flood_reb')
Divide(LHSWorkspace='w55det', RHSWorkspace='wsf1b_flood_reb', OutputWorkspace='w55det', AllowDifferentNumberSpectra=True)

# Sum up the direct beam spectra and do the same for the second workspace.
SumSpectra(InputWorkspace='w55det', OutputWorkspace='w55norm', ListOfWorkspaceIndices='384-414')
ReplaceSpecialValues(InputWorkspace='w55norm', OutputWorkspace='w55norm', NaNValue=0, InfinityValue=0)
ConvertUnits(InputWorkspace='w56', OutputWorkspace='w56', Target='Wavelength', AlignBins=True)
CropWorkspace(InputWorkspace='w56', OutputWorkspace='w56det', StartWorkspaceIndex=5, EndWorkspaceIndex=771)
RebinToWorkspace(WorkspaceToRebin='wsf1b_flood', WorkspaceToMatch='w56det', OutputWorkspace='wsf1b_flood_reb')
Divide(LHSWorkspace='w56det', RHSWorkspace='wsf1b_flood_reb', OutputWorkspace='w56det', AllowDifferentNumberSpectra=True)
SumSpectra(InputWorkspace='w56det', OutputWorkspace='w56norm', ListOfWorkspaceIndices='384-414')
ReplaceSpecialValues(InputWorkspace='w56norm', OutputWorkspace='w56norm', NaNValue=0, InfinityValue=0)

# For a single direct beam I think this will do: s1/s2(h) = 1.95 (30) / 0.1725 (30)
NormaliseByCurrent('w56norm', OutputWorkspace = 'w56norm')
CropWorkspace('w56norm',XMin=1.0, XMax=14.0, OutputWorkspace = 'w56norm')
Rebin('w56norm', Params = 0.05, OutputWorkspace = 'w56norm')

"""
#Scale the direct beam with smaller slits correctly and add them together
Integration(InputWorkspace='w55norm', OutputWorkspace='w55int', RangeLower=1, RangeUpper=14)
Integration(InputWorkspace='w56norm', OutputWorkspace='w56int', RangeLower=1, RangeUpper=14)
Multiply(LHSWorkspace='w55norm', RHSWorkspace='w56int', OutputWorkspace='w55norm')
Divide(LHSWorkspace='w55norm', RHSWorkspace='w55int', OutputWorkspace='w55norm')
MultiplyRange(InputWorkspace='w56norm', OutputWorkspace='w56norm', EndBin=157)
WeightedMean(InputWorkspace1='w55norm', InputWorkspace2='w56norm', OutputWorkspace='DBair03')
"""
