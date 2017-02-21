require('onmt.init')

local cmd = torch.CmdLine()

cmd:text("")
cmd:text("**onmt.translate.lua**")
cmd:text("")


cmd:option('-config', '', [[Read options from this file]])

cmd:text("")
cmd:text("**Data options**")
cmd:text("")

cmd:option('-src', '', [[Source sequence to decode (one line per sequence)]])
cmd:option('-src_domains', '', [[Source side domains]])
cmd:option('-tgt', '', [[True target sequence (optional)]])
cmd:option('-scores', '', [[Source scores sequence (optional) one set of scores per sentence]])
cmd:option('-tgt_domains', '', [[Target side domains]])
cmd:option('-output', 'pred.txt', [[Path to output the predictions (each line will be the decoded sequence]])

onmt.translate.Translator.declareOpts(cmd)

cmd:text("")
cmd:text("**Other options**")
cmd:text("")
cmd:option('-time', false, [[Measure batch translation time]])

onmt.utils.Cuda.declareOpts(cmd)
onmt.utils.Logger.declareOpts(cmd)

local function reportScore(name, scoreTotal, wordsTotal)
  _G.logger:info(name .. " AVG SCORE: %.2f, " .. name .. " PPL: %.2f",
                 scoreTotal / wordsTotal,
                 math.exp(-scoreTotal/wordsTotal))
end

local function main()
  local opt = cmd:parse(arg)

  local requiredOptions = {
    "model",
    "src"
  }

  onmt.utils.Opt.init(opt, requiredOptions)

  _G.logger = onmt.utils.Logger.new(opt.log_file, opt.disable_logs, opt.log_level)

  local srcReader = onmt.utils.FileReader.new(opt.src)
  local srcBatch = {}

  local srcDomainReader
  local tgtDomainReader
  local srcDomainsBatch = {}
  local tgtDomainsBatch = {}

  local inputScoresReader

  if opt.src_domains:len() > 0 then
    srcDomainReader = onmt.utils.FileReader.new(opt.src_domains)
  end
  if opt.tgt_domains:len() > 0 then
    tgtDomainReader = onmt.utils.FileReader.new(opt.tgt_domains)
  end

  local goldReader
  local goldBatch
  local withScores = false

  local withGoldScore = opt.tgt:len() > 0
  if opt.scores:len() > 0 then
    withScores = true
  end

  if withGoldScore then
    goldReader = onmt.utils.FileReader.new(opt.tgt)
    goldBatch = {}
  end
  if withScores then
    inputScoresReader = onmt.utils.FileReader.new(opt.scores)
  end


  local translator = onmt.translate.Translator.new(opt)

  local outFile = io.open(opt.output, 'w')

  local sentId = 1
  local batchId = 1

  local predScoreTotal = 0
  local predWordsTotal = 0
  local goldScoreTotal = 0
  local goldWordsTotal = 0

  local timer
  if opt.time then
    timer = torch.Timer()
    timer:stop()
    timer:reset()
  end

  while true do
    local srcTokens = srcReader:next()
    local goldTokens
    if withGoldScore then
      goldTokens = goldReader:next()
    end

    if withScores then
      inputScoresData = inputScoresReader:next()
    end

    local srcDomain
    local tgtDomain

    if srcDomainReader then
      srcDomain = srcDomainReader:next()[1]
    end
    if tgtDomainReader then
      tgtDomain = tgtDomainReader:next()[1]
    end

    if srcTokens ~= nil then
      table.insert(srcBatch, translator:buildInput(srcTokens, srcDomain, tgtDomain))

      if withGoldScore then
        table.insert(goldBatch, translator:buildInput(goldTokens))
      end
      if withScores then
        local l_inc=0
        local localScoresSent={}
        for l_inc=1,#inputScoresData do
          table.insert(localScoresSent,tonumber(inputScoresData[l_inc]))
        end
        if #localScoresSent > 0 then
          srcBatch[#srcBatch].inputScores=torch.FloatTensor(localScoresSent)
        end
      end
    elseif #srcBatch == 0 then
      break
    end

    if srcTokens == nil or #srcBatch == opt.batch_size then
      if opt.time then
        timer:resume()
      end

      local results = translator:translate(srcBatch, tgtBatch, goldBatch)

      if opt.time then
        timer:stop()
      end

      for b = 1, #results do
        if (#srcBatch[b].words == 0) then
          _G.logger:warning('Line ' .. sentId .. ' is empty.')
          outFile:write('\n')
        else
          _G.logger:info('SENT %d: %s', sentId, translator:buildOutput(srcBatch[b]))

          if withGoldScore then
            _G.logger:info('GOLD %d: %s', sentId, translator:buildOutput(goldBatch[b]), results[b].goldScore)
            _G.logger:info("GOLD SCORE: %.2f", results[b].goldScore)
            goldScoreTotal = goldScoreTotal + results[b].goldScore
            goldWordsTotal = goldWordsTotal + #goldBatch[b]
          end

          for n = 1, #results[b].preds do
            local sentence = translator:buildOutput(results[b].preds[n])

            if n == 1 then
              outFile:write(sentence .. '\n')
              predScoreTotal = predScoreTotal + results[b].preds[n].score
              predWordsTotal = predWordsTotal + #results[b].preds[n].words

              if #results[b].preds > 1 then
                _G.logger:info('')
                _G.logger:info('BEST HYP:')
              end
            end

            if #results[b].preds > 1 then
              _G.logger:info("[%.2f] %s", results[b].preds[n].score, sentence)
            else
              _G.logger:info("PRED %d: %s", sentId, sentence)
              _G.logger:info("PRED SCORE: %.2f", results[b].preds[n].score)
            end
          end
        end

        _G.logger:info('')
        sentId = sentId + 1
      end

      if srcTokens == nil then
        break
      end

      batchId = batchId + 1
      srcBatch = {}
      if withGoldScore then
        goldBatch = {}
      end
      collectgarbage()
    end
  end

  if opt.time then
    local time = timer:time()
    local sentenceCount = sentId-1
    _G.logger:info("Average sentence translation time (in seconds):\n")
    _G.logger:info("avg real\t" .. time.real / sentenceCount .. "\n")
    _G.logger:info("avg user\t" .. time.user / sentenceCount .. "\n")
    _G.logger:info("avg sys\t" .. time.sys / sentenceCount .. "\n")
  end

  reportScore('PRED', predScoreTotal, predWordsTotal)

  if withGoldScore then
    reportScore('GOLD', goldScoreTotal, goldWordsTotal)
  end

  outFile:close()
  _G.logger:shutDown()
end

main()
