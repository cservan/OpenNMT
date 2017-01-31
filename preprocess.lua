require('onmt.init')

local tds = require('tds')

local cmd = torch.CmdLine()

cmd:text("")
cmd:text("preprocess.lua")
cmd:text("")
cmd:text("**Preprocess Options**")
cmd:text("")
cmd:text("")
cmd:option('-config', '', [[Read options from this file]])

cmd:option('-train_src', '', [[Path to the training source data]])
cmd:option('-train_tgt', '', [[Path to the training target data]])
cmd:option('-valid_src', '', [[Path to the validation source data]])
cmd:option('-valid_tgt', '', [[Path to the validation target data]])

cmd:option('-save_data', '', [[Output file for the prepared data]])

cmd:option('-src_vocab_size', '50000', [[Comma-separated list of source vocabularies size: word[,feat1,feat2,...].]])
cmd:option('-tgt_vocab_size', '50000', [[Comma-separated list of target vocabularies size: word[,feat1,feat2,...].]])
cmd:option('-src_vocab', '', [[Path to an existing source vocabulary]])
cmd:option('-tgt_vocab', '', [[Path to an existing target vocabulary]])
cmd:option('-features_vocabs_prefix', '', [[Path prefix to existing features vocabularies]])

cmd:option('-src_seq_length', 50, [[Maximum source sequence length]])
cmd:option('-tgt_seq_length', 50, [[Maximum target sequence length]])
cmd:option('-shuffle', 1, [[Shuffle data]])
cmd:option('-seed', 3435, [[Random seed]])

cmd:option('-report_every', 100000, [[Report status every this many sentences]])

onmt.utils.Logger.declareOpts(cmd)

local opt = cmd:parse(arg)

local function isValid(sent, maxSeqLength)
  return #sent > 0 and #sent <= maxSeqLength
end

local function vecToTensor(vec)
  local t = torch.Tensor(#vec)
  for i, v in pairs(vec) do
    t[i] = v
  end
  return t
end

local function makeData(srcFile, tgtFile, srcDicts, tgtDicts, scoresFile)
  local src = tds.Vec()
  local srcFeatures = tds.Vec()

  local tgt = tds.Vec()
  local tgtFeatures = tds.Vec()

  local sizes = tds.Vec()

  local scoresTable = {}
  local scoresNumber = 0
  
  local count = 0
  local ignored = 0

  local srcReader = onmt.utils.FileReader.new(srcFile)
  local tgtReader = onmt.utils.FileReader.new(tgtFile)
  local scoresReader 
  if scoresFile:len() > 0 then
    scoresReader = onmt.utils.FileReader.new(scoresFile)
  end
  

  while true do
    local scoresStr = nil
    if scoresFile:len() > 0 then
      scoresStr = scoresReader:next()
    end
    local srcTokens = srcReader:next()
    local tgtTokens = tgtReader:next()

    if srcTokens == nil or tgtTokens == nil then
      if srcTokens == nil and tgtTokens ~= nil or srcTokens ~= nil and tgtTokens == nil then
        _G.logger:warning('source and target do not have the same number of sentences')
      end
      break
    end
  if scoresFile:len() > 0 then
    if scoresStr == nil then
        if srcTokens ~= nil and tgtTokens ~= nil then
          print('WARNING: scores and training data do not have the same number of sentences')
          break
        end
    end
  end

    if isValid(srcTokens, opt.src_seq_length) and isValid(tgtTokens, opt.tgt_seq_length) then
      local srcWords, srcFeats = onmt.utils.Features.extract(srcTokens)
      local tgtWords, tgtFeats = onmt.utils.Features.extract(tgtTokens)

      src:insert(srcDicts.words:convertToIdx(srcWords, onmt.Constants.UNK_WORD))
      tgt:insert(tgtDicts.words:convertToIdx(tgtWords,
                                             onmt.Constants.UNK_WORD,
                                             onmt.Constants.BOS_WORD,
                                             onmt.Constants.EOS_WORD))

      if #srcDicts.features > 0 then
        srcFeatures:insert(onmt.utils.Features.generateSource(srcDicts.features, srcFeats, true))
      end
      if #tgtDicts.features > 0 then
        tgtFeatures:insert(onmt.utils.Features.generateTarget(tgtDicts.features, tgtFeats, true))
      end
      if scoresStr ~= nil then
          scoresNumber=#scoresStr
          local l_inc=0
          local localScoresWords={}
          local localScoresSent={}
          for l_inc=1,#scoresStr do
            table.insert(localScoresSent,tonumber(scoresStr[l_inc]))
          end
          local l_inc_wds=0
          -- for l_inc_wds=1,#srcWords do
            -- if localScoresWords == nil then
                -- print ('nil value')
-- --             else
-- --                 print ('TEST BEGIN')
-- --                 print (localScoresWords)
-- --                 print ('TEST END')
            -- end            
            -- table.insert(localScoresSent,localScoresWords)
          -- end
--           print (localScoresSent)
          table.insert(scoresTable,torch.FloatTensor(localScoresSent))
      end
      sizes:insert(#srcWords)
    else
      ignored = ignored + 1
    end

    count = count + 1

    if count % opt.report_every == 0 then
      _G.logger:info('... ' .. count .. ' sentences prepared')
    end
  end

  srcReader:close()
  tgtReader:close()
  if scoresFile:len() > 0 then
    scoresReader:close()
  end

  local function reorderData(perm)
    src = onmt.utils.Table.reorder(src, perm, true)
    tgt = onmt.utils.Table.reorder(tgt, perm, true)

    if #srcDicts.features > 0 then
      srcFeatures = onmt.utils.Table.reorder(srcFeatures, perm, true)
    end
    if #tgtDicts.features > 0 then
      tgtFeatures = onmt.utils.Table.reorder(tgtFeatures, perm, true)
    end
    if #scoresTable > 0 then
        local newTab = {}
        local l_inc = 0
        for l_inc = 1, #scoresTable do
          table.insert(newTab,scoresTable[perm[l_inc]])
        end
     scoresTable=newTab
    end
  end

  if opt.shuffle == 1 then
    _G.logger:info('... shuffling sentences')
    local perm = torch.randperm(#src)
    sizes = onmt.utils.Table.reorder(sizes, perm, true)
    reorderData(perm)
  end

  _G.logger:info('... sorting sentences by size')
  local _, perm = torch.sort(vecToTensor(sizes), true)
  reorderData(perm)

  _G.logger:info('Prepared ' .. #src .. ' sentences (' .. ignored
                   .. ' ignored due to source length > ' .. opt.src_seq_length
                   .. ' or target length > ' .. opt.tgt_seq_length .. ')')

  local srcData = {
    words = src,
    features = srcFeatures
  }

  local tgtData = {
    words = tgt,
    features = tgtFeatures
  }

  return srcData, tgtData, scoresTable
end

local function main()
  local requiredOptions = {
    "train_src",
    "train_tgt",
    "valid_src",
    "valid_tgt",
    "save_data"
  }

  onmt.utils.Opt.init(opt, requiredOptions)

  _G.logger = onmt.utils.Logger.new(opt.log_file, opt.disable_logs, opt.log_level)

  local Vocabulary = onmt.data.Vocabulary

  local data = {}

  data.dicts = {}
  data.dicts.src = Vocabulary.init('source', opt.train_src, opt.src_vocab, opt.src_vocab_size,
                                   opt.features_vocabs_prefix, function(s) return isValid(s, opt.src_seq_length) end)
  data.dicts.tgt = Vocabulary.init('target', opt.train_tgt, opt.tgt_vocab, opt.tgt_vocab_size,
                                   opt.features_vocabs_prefix, function(s) return isValid(s, opt.tgt_seq_length) end)

  _G.logger:info('Preparing training data...')
  data.train = {}
  data.train.src, data.train.tgt, data.train.scores = makeData(opt.train_src, opt.train_tgt,
                                            data.dicts.src, data.dicts.tgt, opt.train_scores)
  _G.logger:info('')

  _G.logger:info('Preparing validation data...')
  data.valid = {}
  data.valid.src, data.valid.tgt, data.valid.scores  = makeData(opt.valid_src, opt.valid_tgt,
                                            data.dicts.src, data.dicts.tgt, opt.valid_scores)
  _G.logger:info('')

  if opt.src_vocab:len() == 0 then
    Vocabulary.save('source', data.dicts.src.words, opt.save_data .. '.src.dict')
  end

  if opt.tgt_vocab:len() == 0 then
    Vocabulary.save('target', data.dicts.tgt.words, opt.save_data .. '.tgt.dict')
  end

  if opt.features_vocabs_prefix:len() == 0 then
    Vocabulary.saveFeatures('source', data.dicts.src.features, opt.save_data)
    Vocabulary.saveFeatures('target', data.dicts.tgt.features, opt.save_data)
  end

  _G.logger:info('Saving data to \'' .. opt.save_data .. '-train.t7\'...')
  torch.save(opt.save_data .. '-train.t7', data, 'binary', false)
  _G.logger:shutDown()
end

main()
