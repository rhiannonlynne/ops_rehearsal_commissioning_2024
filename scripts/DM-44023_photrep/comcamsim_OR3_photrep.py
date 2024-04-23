from lsst.analysis.tools.tasks.reconstructor import reconstructAnalysisTools
from lsst.daf.butler import Butler
from lsst.analysis.tools.interfaces._task import _StandinPlotInfo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import astropy.units as u
from sklearn import linear_model, datasets

def get_photrep_info(mytract, skymap, ccdVisitTable):
    # mytract = 6915
    tract = skymap.generateTract(mytract)
    sp2 = tract.getCtrCoord()
    mytract_did = {'tract': mytract, 'skymap': 'ops_rehersal_prep_2k_v1'}
    taskState, inputData = reconstructAnalysisTools(butler,
                                                    collection=collection,
                                                    label="analyzeMatchedVisitCore",
                                                    dataId=mytract_did, 
                                                    callback=None
    )
    isolated_star_sources = inputData['associatedSources']
    sourcetab = pd.concat(inputData['sourceCatalogs'])
    colnames = ["psfFlux", "psfFluxErr", "psfFlux_flag", "psfFlux_apCorr", "psfFlux_apCorrErr",
            "extendedness", "detect_isPrimary", "deblend_skipped",
            "gaussianFlux", "gaussianFluxErr", "gaussianFlux_flag",
            "localPhotoCalib", "localPhotoCalibErr", "localPhotoCalib_flag"]

    sourcetab = sourcetab[colnames]
    snr_cut = 50.0
    sourcetab = sourcetab[(sourcetab['psfFlux']/sourcetab['psfFluxErr']) > snr_cut]
    # join with the catalog with the associated matches
    joined = pd.merge(isolated_star_sources[["sourceId", "obj_index", "band", "visit", "detector"]], sourcetab, left_on="sourceId", right_index=True)
    joined_ccdinfo = joined.merge(ccdVisitTable, left_on=['visit', 'detector'], right_on=['visitId', 'detector'], how='outer')
    joined_ccdinfo['airmass'] = 1.0/np.cos(np.deg2rad(joined_ccdinfo['zenithDistance']))
    joined_ccdinfo['psfmag'] = (joined_ccdinfo['psfFlux'].values*u.nJy).to(u.ABmag)

    photrep_dict_tmp = {}

    for band in ['r', 'i']:
        photrep_dict_tmp[band] = {}
        tmp_stats_airmass = get_binned_photrep_stats(joined_ccdinfo, 1, 2.2, 0.02, 'airmass', band)
        tmp_stats_seeing = get_binned_photrep_stats(joined_ccdinfo, 0.2, 3.0, 0.02, 'seeing', band)
        photrep_dict_tmp[band]['airmass'] = tmp_stats_airmass
        photrep_dict_tmp[band]['seeing'] = tmp_stats_seeing

    return photrep_dict_tmp


def get_binned_photrep_stats(joined_cat, bin_min, bin_max, binsize, bin_dimension, band):
    bins = np.arange(bin_min, bin_max, binsize)
    bin_cens = np.array(bins)+binsize/2
    # bindim_mag_std = []
    # bindim_mag_count = []
    bindim_data = {}

    pick_band = joined_cat['band_x'] == band
    stars_band = joined_cat[pick_band]
    # Subtract off the mean magnitude for each group, and create a new column:
    stars_band_tmp = stars_band.copy()
    stars_band_tmp['psfmag_norm'] = stars_band_tmp.groupby('obj_index')['psfmag'].transform(lambda x: x - x.mean())

    tmp_mag_std = []
    tmp_mag_count = []
    
    for binmin in bins:
        pick_tmp = (stars_band_tmp[bin_dimension] > binmin) & (stars_band_tmp[bin_dimension] <= binmin+binsize)
        stars_tmp = stars_band_tmp[pick_tmp].groupby('obj_index')
        tmp_mag_std.append(np.nanmedian(stars_tmp.psfmag_norm.aggregate('std')))
        tmp_mag_count.append(np.sum(stars_tmp.psfmag_norm.aggregate('count')))
        # all_tmp_airmass_resid.append(stars_tmp.psfmag_norm.aggregate('std').values)
    
    bindim_data['std'] = tmp_mag_std
    bindim_data['count'] = tmp_mag_count
    bindim_data['bins'] = bin_cens

    return bindim_data


repo = '/repo/embargo'
# collection = 'LSSTComCamSim/runs/nightlyvalidation/20240403/d_2024_03_29/DM-43612'
collection = 'LSSTComCamSim/runs/intermittentcumulativeDRP/20240402_03_04/d_2024_03_29/DM-43865'

butler = Butler(repo, collections=collection)
registry = butler.registry

nImage_refs = list(butler.registry.queryDatasets('deepCoadd_nImage'))

tracts = np.unique([ref.dataId['tract'] for ref in nImage_refs])
print('tracts: ', tracts)

bands = np.unique([ref.dataId['band'] for ref in nImage_refs])
print('bands: ', bands)

# Check which tracts actually have a lot of visit coverage:
print('tract  nvisits')
print('--------------')
for tract in tracts:
    visits = list(butler.registry.queryDatasets('visitSummary', tract=tract, skymap='ops_rehersal_prep_2k_v1', findFirst=True))
    print(tract, len(visits))

skymap = butler.get('skyMap', skymap='ops_rehersal_prep_2k_v1')
ccdVisitTable = butler.get('ccdVisitTable')

photrep_dict_alltracts = {}

# for tract in tracts[2:4]:
for tract in tracts:
    print('Getting tract ', tract)
    photrep_dict_alltracts[tract] = get_photrep_info(tract, skymap, ccdVisitTable)

# import pdb; pdb.set_trace()

# photrep_dict_alltracts[tract]['r']['seeing']['std'] (or 'count', or 'bins')

seeing_std_r = [photrep_dict_alltracts[key]['r']['seeing']['std'] for key in photrep_dict_alltracts.keys()]
seeing_ct_r = [photrep_dict_alltracts[key]['r']['seeing']['count'] for key in photrep_dict_alltracts.keys()]
seeing_std_i = [photrep_dict_alltracts[key]['i']['seeing']['std'] for key in photrep_dict_alltracts.keys()]
seeing_ct_i = [photrep_dict_alltracts[key]['i']['seeing']['count'] for key in photrep_dict_alltracts.keys()]
keylist = list(photrep_dict_alltracts.keys())
key0 = keylist[0]
seeing_bins = photrep_dict_alltracts[key0]['r']['seeing']['bins']

airmass_std_r = [photrep_dict_alltracts[key]['r']['airmass']['std'] for key in photrep_dict_alltracts.keys()]
airmass_ct_r = [photrep_dict_alltracts[key]['r']['airmass']['count'] for key in photrep_dict_alltracts.keys()]
airmass_std_i = [photrep_dict_alltracts[key]['i']['airmass']['std'] for key in photrep_dict_alltracts.keys()]
airmass_ct_i = [photrep_dict_alltracts[key]['i']['airmass']['count'] for key in photrep_dict_alltracts.keys()]
airmass_bins = photrep_dict_alltracts[key0]['r']['airmass']['bins']

params = {'axes.labelsize': 24,
          'font.size': 20,
          'legend.fontsize': 9,
          'xtick.major.width': 3,
          'xtick.minor.width': 2,
          'xtick.major.size': 12,
          'xtick.minor.size': 6,
          'xtick.direction': 'in',
          'xtick.top': True,
          'lines.linewidth': 3,
          'axes.linewidth': 3,
          'axes.labelweight': 3,
          'axes.titleweight': 3,
          'ytick.major.width': 3,
          'ytick.minor.width': 2,
          'ytick.major.size': 12,
          'ytick.minor.size': 6,
          'ytick.direction': 'in',
          'ytick.right': True,
          'figure.figsize': [9, 7],
          'figure.facecolor': 'White'}
plt.rcParams.update(params)

nvisits_min = 20

# for seeing_stats in seeing_std_r:
for i in range(len(seeing_std_r)):
    seeing_stats = seeing_std_r[i]
    seeing_counts = seeing_ct_r[i]
    okbins = (np.array(seeing_counts) > nvisits_min)
    # import pdb; pdb.set_trace()
    tract_number = keylist[i]

    randcolor = (np.random.random(), np.random.random(), np.random.random())

    plt.plot(seeing_bins[okbins], 1000.0*np.array(seeing_stats)[okbins], 'o',
             color=randcolor, label=str(tract_number))

plt.xlabel('seeing')
plt.ylabel('photometric repeatability (mmag)')
plt.title('r-band; S/N>50, nvisits>'+str(nvisits_min))
plt.minorticks_on()
plt.xlim(0.38, 2.18)
plt.ylim(2, 18)
plt.legend(ncol=4)

plt.savefig('comcamsim_photrep_seeing_rband_alltracts.png')
plt.close()

for i in range(len(seeing_std_i)):
    seeing_stats = seeing_std_i[i]
    seeing_counts = seeing_ct_i[i]
    okbins = (np.array(seeing_counts) > nvisits_min)
    # import pdb; pdb.set_trace()
    tract_number = keylist[i]

    randcolor = (np.random.random(), np.random.random(), np.random.random())

    plt.plot(seeing_bins[okbins], 1000.0*np.array(seeing_stats)[okbins], 'o',
             color=randcolor, label=str(tract_number))

plt.xlabel('seeing')
plt.ylabel('photometric repeatability (mmag)')
plt.title('i-band; S/N>50, nvisits>'+str(nvisits_min))
plt.minorticks_on()
plt.xlim(0.38, 2.18)
plt.ylim(2, 18)
plt.legend(ncol=4)

plt.savefig('comcamsim_photrep_seeing_iband_alltracts.png')
plt.close()

for i in range(len(airmass_std_r)):
    airmass_stats = airmass_std_r[i]
    airmass_counts = airmass_ct_r[i]
    okbins = (np.array(airmass_counts) > nvisits_min)
    tract_number = keylist[i]

    randcolor = (np.random.random(), np.random.random(), np.random.random())

    plt.plot(airmass_bins[okbins], 1000.0*np.array(airmass_stats)[okbins], 'o',
             color=randcolor, label=str(tract_number))

plt.xlabel('airmass')
plt.ylabel('photometric repeatability (mmag)')
plt.title('r-band; S/N>50, nvisits>'+str(nvisits_min))
plt.minorticks_on()
plt.xlim(1.0, 2.3)
plt.ylim(2, 18)
plt.legend(ncol=4)

plt.savefig('comcamsim_photrep_airmass_rband_alltracts.png')
plt.close()

for i in range(len(airmass_std_i)):
    airmass_stats = airmass_std_i[i]
    airmass_counts = airmass_ct_i[i]
    okbins = (np.array(airmass_counts) > nvisits_min)
    tract_number = keylist[i]

    randcolor = (np.random.random(), np.random.random(), np.random.random())

    plt.plot(airmass_bins[okbins], 1000.0*np.array(airmass_stats)[okbins], 'o',
             color=randcolor, label=str(tract_number))

plt.xlabel('airmass')
plt.ylabel('photometric repeatability (mmag)')
plt.title('i-band; S/N>50, nvisits>'+str(nvisits_min))
plt.minorticks_on()
plt.xlim(1.0, 2.3)
plt.ylim(2, 18)
plt.legend(ncol=4)

plt.savefig('comcamsim_photrep_airmass_iband_alltracts.png')
plt.close()

