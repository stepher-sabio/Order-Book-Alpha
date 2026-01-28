#include <databento/dbn.hpp>
#include <databento/record.hpp>
#include <databento/dbn_decoder.hpp>
#include <databento/datetime.hpp>
#include <databento/file_stream.hpp>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/writer.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <set>

// Extract Apple data to Parquet format
void extractAppleData(const std::vector<std::string>& files, 
                      const std::string& basePath,
                      uint32_t apple_id) {
    
    // Define Arrow schema
    auto schema = arrow::schema({
        arrow::field("timestamp", arrow::int64()),
        arrow::field("instrument_id", arrow::uint32()),
        arrow::field("bid_px_0", arrow::int64()),
        arrow::field("ask_px_0", arrow::int64()),
        arrow::field("bid_sz_0", arrow::uint32()),
        arrow::field("ask_sz_0", arrow::uint32()),
        arrow::field("bid_ct_0", arrow::uint32()),
        arrow::field("ask_ct_0", arrow::uint32()),
        arrow::field("bid_px_1", arrow::int64()),
        arrow::field("ask_px_1", arrow::int64()),
        arrow::field("bid_sz_1", arrow::uint32()),
        arrow::field("ask_sz_1", arrow::uint32()),
        arrow::field("bid_px_2", arrow::int64()),
        arrow::field("ask_px_2", arrow::int64()),
        arrow::field("bid_sz_2", arrow::uint32()),
        arrow::field("ask_sz_2", arrow::uint32()),
        arrow::field("spread", arrow::int64()),
        arrow::field("mid_price", arrow::float64()),
        arrow::field("action", arrow::utf8()),
        arrow::field("side", arrow::utf8()),
        arrow::field("depth", arrow::uint8()),
        arrow::field("sequence", arrow::uint32())
    });

    // Create builders for each column
    arrow::Int64Builder timestamp_builder;
    arrow::UInt32Builder instrument_id_builder;
    arrow::Int64Builder bid_px_0_builder, ask_px_0_builder;
    arrow::UInt32Builder bid_sz_0_builder, ask_sz_0_builder;
    arrow::UInt32Builder bid_ct_0_builder, ask_ct_0_builder;
    arrow::Int64Builder bid_px_1_builder, ask_px_1_builder;
    arrow::UInt32Builder bid_sz_1_builder, ask_sz_1_builder;
    arrow::Int64Builder bid_px_2_builder, ask_px_2_builder;
    arrow::UInt32Builder bid_sz_2_builder, ask_sz_2_builder;
    arrow::Int64Builder spread_builder;
    arrow::DoubleBuilder mid_price_builder;
    arrow::StringBuilder action_builder, side_builder;
    arrow::UInt8Builder depth_builder;
    arrow::UInt32Builder sequence_builder;

    int total_apple_records = 0;
    
    for (const auto& file : files) {
        std::string filePath = basePath + file;
        std::cout << "Processing: " << file << "... " << std::flush;
        
        try {
            auto file_stream = std::make_unique<databento::InFileStream>(filePath);
            databento::DbnDecoder decoder{nullptr, std::move(file_stream)};
            
            auto meta = decoder.DecodeMetadata();
            
            const databento::Record* rec;
            int file_apple_count = 0;
            
            while ((rec = decoder.DecodeRecord()) != nullptr) {
                if (rec->RType() == databento::RType::Mbp10) {
                    const auto& mbp = rec->Get<databento::Mbp10Msg>();
                    
                    // Only process Apple records
                    if (mbp.hd.instrument_id == apple_id) {
                        // Calculate metrics
                        int64_t spread = mbp.levels[0].ask_px - mbp.levels[0].bid_px;
                        double mid_price = (mbp.levels[0].bid_px + mbp.levels[0].ask_px) / 2.0;

                        // Append to builders (convert timestamp to nanoseconds)
                        auto ts_nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            mbp.hd.ts_event.time_since_epoch()).count();
                        timestamp_builder.Append(ts_nanos);
                        instrument_id_builder.Append(mbp.hd.instrument_id);
                        
                        bid_px_0_builder.Append(mbp.levels[0].bid_px);
                        ask_px_0_builder.Append(mbp.levels[0].ask_px);
                        bid_sz_0_builder.Append(mbp.levels[0].bid_sz);
                        ask_sz_0_builder.Append(mbp.levels[0].ask_sz);
                        bid_ct_0_builder.Append(mbp.levels[0].bid_ct);
                        ask_ct_0_builder.Append(mbp.levels[0].ask_ct);
                        
                        bid_px_1_builder.Append(mbp.levels[1].bid_px);
                        ask_px_1_builder.Append(mbp.levels[1].ask_px);
                        bid_sz_1_builder.Append(mbp.levels[1].bid_sz);
                        ask_sz_1_builder.Append(mbp.levels[1].ask_sz);
                        
                        bid_px_2_builder.Append(mbp.levels[2].bid_px);
                        ask_px_2_builder.Append(mbp.levels[2].ask_px);
                        bid_sz_2_builder.Append(mbp.levels[2].bid_sz);
                        ask_sz_2_builder.Append(mbp.levels[2].ask_sz);
                        
                        spread_builder.Append(spread);
                        mid_price_builder.Append(mid_price);
                        
                        action_builder.Append(std::string(1, static_cast<char>(mbp.action)));
                        side_builder.Append(std::string(1, static_cast<char>(mbp.side)));
                        
                        depth_builder.Append(mbp.depth);
                        sequence_builder.Append(mbp.sequence);
                        
                        file_apple_count++;
                        total_apple_records++;
                    }
                }
            }
            
            std::cout << file_apple_count << " Apple records\n";
            
        } catch (const std::exception& e) {
            std::cout << "ERROR: " << e.what() << "\n";
        }
    }

    std::cout << "\nBuilding Parquet file...\n";

    // Finish building arrays
    std::shared_ptr<arrow::Array> timestamp_array, instrument_id_array;
    std::shared_ptr<arrow::Array> bid_px_0_array, ask_px_0_array;
    std::shared_ptr<arrow::Array> bid_sz_0_array, ask_sz_0_array;
    std::shared_ptr<arrow::Array> bid_ct_0_array, ask_ct_0_array;
    std::shared_ptr<arrow::Array> bid_px_1_array, ask_px_1_array;
    std::shared_ptr<arrow::Array> bid_sz_1_array, ask_sz_1_array;
    std::shared_ptr<arrow::Array> bid_px_2_array, ask_px_2_array;
    std::shared_ptr<arrow::Array> bid_sz_2_array, ask_sz_2_array;
    std::shared_ptr<arrow::Array> spread_array, mid_price_array;
    std::shared_ptr<arrow::Array> action_array, side_array;
    std::shared_ptr<arrow::Array> depth_array, sequence_array;

    timestamp_builder.Finish(&timestamp_array);
    instrument_id_builder.Finish(&instrument_id_array);
    bid_px_0_builder.Finish(&bid_px_0_array);
    ask_px_0_builder.Finish(&ask_px_0_array);
    bid_sz_0_builder.Finish(&bid_sz_0_array);
    ask_sz_0_builder.Finish(&ask_sz_0_array);
    bid_ct_0_builder.Finish(&bid_ct_0_array);
    ask_ct_0_builder.Finish(&ask_ct_0_array);
    bid_px_1_builder.Finish(&bid_px_1_array);
    ask_px_1_builder.Finish(&ask_px_1_array);
    bid_sz_1_builder.Finish(&bid_sz_1_array);
    ask_sz_1_builder.Finish(&ask_sz_1_array);
    bid_px_2_builder.Finish(&bid_px_2_array);
    ask_px_2_builder.Finish(&ask_px_2_array);
    bid_sz_2_builder.Finish(&bid_sz_2_array);
    ask_sz_2_builder.Finish(&ask_sz_2_array);
    spread_builder.Finish(&spread_array);
    mid_price_builder.Finish(&mid_price_array);
    action_builder.Finish(&action_array);
    side_builder.Finish(&side_array);
    depth_builder.Finish(&depth_array);
    sequence_builder.Finish(&sequence_array);

    // Create table
    auto table = arrow::Table::Make(schema, {
        timestamp_array, instrument_id_array,
        bid_px_0_array, ask_px_0_array, bid_sz_0_array, ask_sz_0_array,
        bid_ct_0_array, ask_ct_0_array,
        bid_px_1_array, ask_px_1_array, bid_sz_1_array, ask_sz_1_array,
        bid_px_2_array, ask_px_2_array, bid_sz_2_array, ask_sz_2_array,
        spread_array, mid_price_array,
        action_array, side_array, depth_array, sequence_array
    });

    // Write to Parquet file
    std::shared_ptr<arrow::io::FileOutputStream> outfile;
    PARQUET_ASSIGN_OR_THROW(
        outfile,
        arrow::io::FileOutputStream::Open("apple_data.parquet"));

    PARQUET_THROW_NOT_OK(
        parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), outfile, 1000000));

    std::cout << "\n✅ EXTRACTION COMPLETE!\n";
    std::cout << "Total Apple (AAPL) records: " << total_apple_records << "\n";
    std::cout << "Output file: apple_data.parquet\n";
}

int main() {
    const std::string basePath = "../Order Book Alpha /XNAS-20251003-VP9PGU8PQJ/";
    
    std::vector<std::string> files = {
        "xnas-itch-20250703.mbp-10.dbn.zst",
        "xnas-itch-20250704.mbp-10.dbn.zst",
        "xnas-itch-20250705.mbp-10.dbn.zst",
        "xnas-itch-20250706.mbp-10.dbn.zst",
        "xnas-itch-20250707.mbp-10.dbn.zst",
        "xnas-itch-20250708.mbp-10.dbn.zst",
        "xnas-itch-20250709.mbp-10.dbn.zst",
        "xnas-itch-20250710.mbp-10.dbn.zst",
        "xnas-itch-20250711.mbp-10.dbn.zst",
        "xnas-itch-20250712.mbp-10.dbn.zst",
        "xnas-itch-20250713.mbp-10.dbn.zst",
        "xnas-itch-20250714.mbp-10.dbn.zst",
        "xnas-itch-20250715.mbp-10.dbn.zst",
        "xnas-itch-20250716.mbp-10.dbn.zst",
        "xnas-itch-20250717.mbp-10.dbn.zst",
        "xnas-itch-20250718.mbp-10.dbn.zst",
        "xnas-itch-20250719.mbp-10.dbn.zst",
        "xnas-itch-20250720.mbp-10.dbn.zst",
        "xnas-itch-20250721.mbp-10.dbn.zst",
        "xnas-itch-20250722.mbp-10.dbn.zst",
        "xnas-itch-20250723.mbp-10.dbn.zst",
        "xnas-itch-20250724.mbp-10.dbn.zst",
        "xnas-itch-20250725.mbp-10.dbn.zst",
        "xnas-itch-20250726.mbp-10.dbn.zst",
        "xnas-itch-20250727.mbp-10.dbn.zst",
        "xnas-itch-20250728.mbp-10.dbn.zst",
        "xnas-itch-20250729.mbp-10.dbn.zst",
        "xnas-itch-20250730.mbp-10.dbn.zst",
        "xnas-itch-20250731.mbp-10.dbn.zst",
        "xnas-itch-20250801.mbp-10.dbn.zst",
        "xnas-itch-20250802.mbp-10.dbn.zst",
        "xnas-itch-20250803.mbp-10.dbn.zst",
        "xnas-itch-20250804.mbp-10.dbn.zst",
        "xnas-itch-20250805.mbp-10.dbn.zst",
        "xnas-itch-20250806.mbp-10.dbn.zst",
        "xnas-itch-20250807.mbp-10.dbn.zst",
        "xnas-itch-20250808.mbp-10.dbn.zst",
        "xnas-itch-20250809.mbp-10.dbn.zst",
        "xnas-itch-20250810.mbp-10.dbn.zst",
        "xnas-itch-20250811.mbp-10.dbn.zst",
        "xnas-itch-20250812.mbp-10.dbn.zst",
        "xnas-itch-20250813.mbp-10.dbn.zst",
        "xnas-itch-20250814.mbp-10.dbn.zst",
        "xnas-itch-20250815.mbp-10.dbn.zst",
        "xnas-itch-20250816.mbp-10.dbn.zst",
        "xnas-itch-20250817.mbp-10.dbn.zst",
        "xnas-itch-20250818.mbp-10.dbn.zst",
        "xnas-itch-20250819.mbp-10.dbn.zst",
        "xnas-itch-20250820.mbp-10.dbn.zst",
        "xnas-itch-20250821.mbp-10.dbn.zst",
        "xnas-itch-20250822.mbp-10.dbn.zst",
        "xnas-itch-20250823.mbp-10.dbn.zst",
        "xnas-itch-20250824.mbp-10.dbn.zst",
        "xnas-itch-20250825.mbp-10.dbn.zst",
        "xnas-itch-20250826.mbp-10.dbn.zst",
        "xnas-itch-20250827.mbp-10.dbn.zst",
        "xnas-itch-20250828.mbp-10.dbn.zst",
        "xnas-itch-20250829.mbp-10.dbn.zst",
        "xnas-itch-20250830.mbp-10.dbn.zst",
        "xnas-itch-20250831.mbp-10.dbn.zst",
        "xnas-itch-20250901.mbp-10.dbn.zst",
        "xnas-itch-20250902.mbp-10.dbn.zst",
        "xnas-itch-20250903.mbp-10.dbn.zst",
        "xnas-itch-20250904.mbp-10.dbn.zst",
        "xnas-itch-20250905.mbp-10.dbn.zst",
        "xnas-itch-20250906.mbp-10.dbn.zst",
        "xnas-itch-20250907.mbp-10.dbn.zst",
        "xnas-itch-20250908.mbp-10.dbn.zst",
        "xnas-itch-20250909.mbp-10.dbn.zst",
        "xnas-itch-20250910.mbp-10.dbn.zst",
        "xnas-itch-20250911.mbp-10.dbn.zst",
        "xnas-itch-20250912.mbp-10.dbn.zst",
        "xnas-itch-20250913.mbp-10.dbn.zst",
        "xnas-itch-20250914.mbp-10.dbn.zst",
        "xnas-itch-20250915.mbp-10.dbn.zst",
        "xnas-itch-20250916.mbp-10.dbn.zst",
        "xnas-itch-20250917.mbp-10.dbn.zst",
        "xnas-itch-20250918.mbp-10.dbn.zst",
        "xnas-itch-20250919.mbp-10.dbn.zst",
        "xnas-itch-20250920.mbp-10.dbn.zst",
        "xnas-itch-20250921.mbp-10.dbn.zst",
        "xnas-itch-20250922.mbp-10.dbn.zst",
        "xnas-itch-20250923.mbp-10.dbn.zst",
        "xnas-itch-20250924.mbp-10.dbn.zst",
        "xnas-itch-20250925.mbp-10.dbn.zst",
        "xnas-itch-20250926.mbp-10.dbn.zst",
        "xnas-itch-20250927.mbp-10.dbn.zst",
        "xnas-itch-20250928.mbp-10.dbn.zst",
        "xnas-itch-20250929.mbp-10.dbn.zst",
        "xnas-itch-20250930.mbp-10.dbn.zst",
        "xnas-itch-20251001.mbp-10.dbn.zst",
        "xnas-itch-20251002.mbp-10.dbn.zst"
    };

    // Apple's instrument_id is 38
    uint32_t APPLE_ID = 38;
    
    std::cout << "\n🍎 Extracting Apple (instrument_id: " << APPLE_ID << ") data to Parquet...\n\n";
    extractAppleData(files, basePath, APPLE_ID);

    return 0;
}