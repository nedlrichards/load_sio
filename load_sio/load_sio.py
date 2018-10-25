import numpy as np
import os


def load_allChannels(fileName, channelNames=None):
    """ Load all data from an sio file
    """
    return load_selection(fileName, -1, 0, -1, channelNames)


def load_selection(fileName, sampleStart, numSamples,
                   channels, channelNames=None):
    """ Load a section of data that can be a selection of time and/or channels.

    If either sampleStart or channels is -1 all of the data is loaded in that
    dimension.
    """
    fileInfo = load_header(fileName, channelNames=channelNames)

    memoryMap = createSIOMap(fileInfo)

    return loadFromMap(memoryMap, fileInfo, sampleStart, numSamples, channels)


def loadFromMap(sioMap, fileInfo, sampleStart, numSamples, channels):
    """ load data from memory map and return as library with header data """
    recPerChan = np.int(fileInfo['numRecords'] / fileInfo['numChannels'])
    samPerRec = np.int(fileInfo['recordLength'] / fileInfo['bytesPerPoint'])
   # Create the time load indicies
    if sampleStart < 0:
        recStartI = 0
        recEndI = recPerChan
        startOffsetI = 0
        endOffsetI = 0
    elif (sampleStart + numSamples) > fileInfo['samplesPerChannel']:
        raise NameError('Index range exceeds file bounds')
    else:
        recStartI, startOffsetI = divmod(sampleStart, samPerRec)
        # Load 1 extra record to take care of remainder
        recEndI = int((sampleStart + numSamples) / samPerRec) + 1
        endOffsetI = startOffsetI + numSamples

    channels = np.array(channels)
    # Create the channel load indicies
    if np.any(np.less(channels,  0)):
        channels = range(fileInfo['numChannels'])
    # Check for requested channels that are not in channel range
    elif np.any(channels >= fileInfo['numChannels']):
        raise NameError('Requested channels that do not exist')
    else:
        pass

    data = np.array(sioMap[recStartI:recEndI, channels, :], ndmin=2)

    # Different concatenation for when channels is a single number or an array
    if data.ndim == 3:
        data = np.concatenate(data, 1)
    elif data.ndim == 2:
        data = np.concatenate(data)

    # Force data to have 2 dimensions, makes 1 x N for 1 channel case
    data = np.array(data, ndmin=2)

    # Remove any offset less than a block length
    if endOffsetI:
        data = data[:, startOffsetI: endOffsetI]
    else:
        data = data[:, startOffsetI:-1]

    data = data.T
    # return loaded data as a dictionary with the header information
    fileContents = fileInfo
    fileContents['data'] = np.squeeze(data)
    return fileContents


def createSIOMap(fileInfo):
    """ create a memory map for loading SIO data """
    fileName = fileInfo['fileName']
    recPerChan = np.int(fileInfo['numRecords'] / fileInfo['numChannels'])
    samPerRec = np.int(fileInfo['recordLength'] / fileInfo['bytesPerPoint'])

    f = np.memmap(fileName, dtype=fileInfo['recordType'], mode='r',
                  offset=fileInfo['recordType'].itemsize,
                  shape=(recPerChan, int(fileInfo['numChannels'])))

    return f


def load_header(fileName, channelNames=None):
    """ Load header from sio file

    channelName is an optional list of strings of the name of each channel
    """
    # Assume little endian
    isBigEndian = False
    dtype_header = np.dtype(('<i4', 8))

    with open(fileName, "rb") as f:
        header = np.fromfile(f, dtype_header, 1)
        header = header.flatten()

        # read makes sense for strings
        name = f.read(24)
        com = f.read(72)

        # after read check endian
        check = header[-1]

        if check != 32677:
            if check.byteswap() == 32677:
                isBigEndian = True
            else:
                raise NameError('Unknown byte order')

        # isReal flag does not seem to do anything in sioread.m

        fieldNames = ['fid', 'numRecords', 'recordLength',
                      'numChannels', 'bytesPerPoint', 'isReal',
                      'samplesPerChannel', 'endianCheck']

        if isBigEndian:
            header = header.byteswap()

        # save meta data in a dictionary
        fileInfo = dict(zip(fieldNames, header))
        fileInfo['fileName'] = fileName
        fileInfo['readName'] = name
        fileInfo['comment'] = com

        # Check for user supplies channel names, if not create index as names

        if channelNames is None:
            channelNames = np.arange(fileInfo['numChannels'])
        # elif len(channelNames) != fileInfo['numChannels']:
        #    raise ValueError
        fileInfo['channelName'] = channelNames

        if fileInfo['bytesPerPoint'] == 4:
            dataType = np.dtype('<f4')
        elif fileInfo['bytesPerPoint'] == 2:
            dataType = np.dtype('<i2')
        else:
            raise NameError('unknown data type')

        if isBigEndian:
            dataType = dataType.newbyteorder()

        # read data as a series of records
        samPerRec = fileInfo['recordLength'] / fileInfo['bytesPerPoint']
        recordType = np.dtype((dataType, np.int(samPerRec)))

        fileInfo['recordType'] = recordType
        return fileInfo
