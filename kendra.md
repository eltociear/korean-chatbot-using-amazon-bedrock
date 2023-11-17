# Kendra 성능 향상 방법

Kendra의 [Retrieve API](https://docs.aws.amazon.com/kendra/latest/APIReference/API_Retrieve.html)를 이용합니다.

Type의 종류에는 "DOCUMENT", "QUESTION_ANSWER", "ANSWER"가 있습니다.

Type은 query결과에서 "Format"으로 구분할 수 있습니다.

```text
[
   {
      "Id":"74719041-8126-473c-92f1-929fdc520138-188b319d-552f-4ff4-a7d5-8cbcd21dbea8",
      "Type":"QUESTION_ANSWER",
      "Format":"TEXT",
      "AdditionalAttributes":[
         {
            "Key":"QuestionText",
            "ValueType":"TEXT_WITH_HIGHLIGHTS_VALUE",
            "Value":{
               "TextWithHighlightsValue":{
                  "Text":"How many free clinics are in Spokane WA?",
                  "Highlights":[
                     {
                        "BeginOffset":4,
                        "EndOffset":8,
                        "TopAnswer":false,
                        "Type":"STANDARD"
                     },
                     {
                        "BeginOffset":9,
                        "EndOffset":13,
                        "TopAnswer":false,
                        "Type":"STANDARD"
                     },
                     {
                        "BeginOffset":14,
                        "EndOffset":21,
                        "TopAnswer":false,
                        "Type":"STANDARD"
                     },
                     {
                        "BeginOffset":29,
                        "EndOffset":36,
                        "TopAnswer":false,
                        "Type":"STANDARD"
                     },
                     {
                        "BeginOffset":37,
                        "EndOffset":39,
                        "TopAnswer":false,
                        "Type":"STANDARD"
                     }
                  ]
               }
            }
         },
         {
            "Key":"AnswerText",
            "ValueType":"TEXT_WITH_HIGHLIGHTS_VALUE",
            "Value":{
               "TextWithHighlightsValue":{
                  "Text":"13",
                  "Highlights":[
                     
                  ]
               }
            }
         }
      ],
      "DocumentId":"c24c0fe9cbdfa412ac58d1b5fc07dfd4afd21cbd0f71df499f305296d985a8c9a91f1b2c-e28b-4d13-8b01-8a33be5fc126",
      "DocumentTitle":{
         "Text":""
      },
      "DocumentExcerpt":{
         "Text":"13",
         "Highlights":[
            {
               "BeginOffset":0,
               "EndOffset":2,
               "TopAnswer":false,
               "Type":"STANDARD"
            }
         ]
      },
      "DocumentURI":"https://www.freeclinics.com/",
      "DocumentAttributes":[
         {
            "Key":"_source_uri",
            "Value":{
               "StringValue":"https://www.freeclinics.com/"
            }
         }
      ],
      "ScoreAttributes":{
         "ScoreConfidence":"VERY_HIGH"
      },
      "FeedbackToken":"AYADeN-jZ9DvGVP9n00b4d48LrsAXwABABVhd3MtY3J5cHRvLXB1YmxpYy1rZXkAREF6ajZZVkJ3M3B4dXZEMGRJZitQaEEzUWNVZkE3TDVBbjNEOCs1bE1aRm1hN1M3a0N3cjNiMzZRR2hPcTloeVJ1QT09AAEAB2F3cy1rbXMAS2Fybjphd3M6a21zOnVzLXdlc3QtMjoxNDk0MDA5NDM5NTk6a2V5LzUyN2YwMjRhLTUyMDktNDI4NC1iOTYwLTJhMjYxMzQxNWNkNgC4AQIBAHhoFIrDBc0sA_W0qqJvieboGJWYBK_hEm739PftPtfwZwEP6KAczOsL3xpUp6oizSAgAAAAfjB8BgkqhkiG9w0BBwagbzBtAgEAMGgGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMT_QgV_BMu5l49EZAAgEQgDuTOkP1QJbt85KZ4FDF438i0upluDZq_Rf3L8H9PqkLQOSgUAyyy9hqEmMOZUcGqvBNc_ekw4pbMRy5ZAIAAAAADAAAEAAAAAAAAAAAAAAAAAB3LYEFCQuAEb8NnKHSk1eT_____wAAAAEAAAAAAAAAAAAAAAEAAAF2QLnDNTO_Ma1EGreEOHC8YG5ijJ7jLblLE4CbyAY9ueJHKBTQ-Rf2A_pD9hpXTuyP6Ho84IIlScm7IhFUomBUSZMD_qrc0qnvlrCjgXwJ_AM0MJKmqBkMPNvivFnfZ9xl-dFyFX1sdzq0_LUE4KgLZpjQiSU0b_PFJw2zN8P6JSJb9Fz84fbWu1_nzrVrqCj5dDpMLDNLgC3f6pTS4IqmJqMsj6BbGcdsvLIzVA2XaAGYS8CNv9pu5Hz63yrh6hG4UHWJwdhIcPZG7z7BayFjravsKjw101PJnzUKSIfiZlRnoqm-Bbff-ieECV-vZ_1vtskbHhmsZ4WlKTcpD5QGMrElbk7WMbdPf8gmGQfC8SMrR-ixO7d856LIsoTx9i6VcN91GxEKcYtsXY4J0w6G4aL8-tj1iS7zwIsxHimIsuAHM4u5SmHmI_oJ25pR-7TA2K34GVv9VhYydG8JsBbjGV-mPpg6ORE4bNkhRL38f1pfHEXNlv79F9b8UP93MLfj6lZT25tPAGcwZQIxAK_GncCyOyt2NLdszY-Oc2Qchpo2CCTjj25a5wyzYv4JObw591oaxZeSVbA_Mq2v2gIwf6xq0c5vYGait9J9mnI2FMtEJ3rI2DRld30IRmWWNES54XOxciMd5J_YxJGkfghX.74719041-8126-473c-92f1-929fdc520138-188b319d-552f-4ff4-a7d5-8cbcd21dbea8"
   }
]
```

## FAQ

[FAQ-Kendra](https://github.com/aws-samples/enterprise-search-with-amazon-kendra-workshop/blob/master/Part%202%20-%20Adding%20a%20FAQ.md)를 참조합니다.

[kendra-faq-refresher](https://github.com/aws-samples/amazon-kendra-faq-refresher/tree/main)를 참조하여 FAQ를 Kendra 검색 결과로 활용할 수 있습니다.


## Reference

[Retrieve API](https://docs.aws.amazon.com/kendra/latest/APIReference/API_Retrieve.html)

